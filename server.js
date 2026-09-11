require("dotenv").config();

const express = require("express");
const Anthropic = require("@anthropic-ai/sdk");
const axios = require("axios");

const app = express();
app.use(express.json());
app.use(express.static("public"));

const client = new Anthropic({
  apiKey: process.env.ANTHROPIC_API_KEY,
});

const GOOGLE_API_KEY = process.env.GOOGLE_API_KEY;

// Fallback only — used if the browser's geolocation fails or is denied.
// No longer assumes Osaka; leave null and require real coordinates instead.
const DEFAULT_LAT = null;
const DEFAULT_LNG = null;

// Explicit dollar amounts — these are a genuine fixed request ("under $15"
// means under $15 anywhere), so they stay mapped to Google's absolute
// price_level scale rather than treated as relative.
const PRICE_BUDGET_MAP = {
  "under 10": 1, "under $10": 1,
  "10-15": 1, "$10-15": 1, "10-20": 1, "$10-20": 1, "around $10": 1, "around $15": 1,
  "10-30": 2, "$10-30": 2, "around $20": 2, "around $25": 2, "around $30": 2,
  "30-60": 3, "$30-60": 3, "around $40": 3, "around $50": 3,
  "60+": 4, "$60+": 4,
};

function parsePriceCeiling(priceStr) {
  if (!priceStr) return null;
  const lower = priceStr.toLowerCase();
  for (const [key, level] of Object.entries(PRICE_BUDGET_MAP)) {
    if (lower.includes(key)) return level;
  }
  return null;
}

// Vague/relative price words ("cheap", "upscale") don't name a dollar figure —
// what counts as "cheap" depends entirely on what's actually available near
// the user (see filterByRelativePrice), unlike the fixed dollar phrases above.
const RELATIVE_PRICE_WORDS = {
  low: ["cheap", "budget", "affordable", "inexpensive", "cheapest", "cheaply"],
  high: ["expensive", "upscale", "pricey", "fine dining", "fancy", "high-end", "splurge"],
  mid: ["moderate", "mid-range", "mid range", "average priced", "reasonably priced"],
};
const NO_PRICE_FILTER_WORDS = ["no limit", "unlimited", "no budget"];

function parseRelativePriceRank(priceStr) {
  if (!priceStr) return null;
  const lower = priceStr.toLowerCase();
  if (NO_PRICE_FILTER_WORDS.some(w => lower.includes(w))) return "none";
  for (const [rank, words] of Object.entries(RELATIVE_PRICE_WORDS)) {
    if (words.some(w => lower.includes(w))) return rank;
  }
  return null;
}

const PROXIMITY_SPEED_KMH = { walking: 5, biking: 15, driving: 25 };
const PROXIMITY_DEFAULT_MINUTES = { walking: 15, biking: 15, driving: 10 };
const DEFAULT_RADIUS_KM = 5; // nothing said about proximity at all
const NEAR_ME_RADIUS_KM = 1.5; // "near me" / "close by" said explicitly, but no mode or time given

// Converts a parsed proximity intent ("walking distance", "10 min bike ride", etc.)
// into an actual search radius. Claude only classifies mode/minutes from the query
// text — the km math is done here so it's exact rather than model best-effort.
function computeRadiusKm(proximity) {
  if (!proximity) {
    return DEFAULT_RADIUS_KM;
  }
  if (!proximity.mode && !proximity.minutes) {
    return NEAR_ME_RADIUS_KM;
  }
  const mode = proximity.mode || "walking";
  const speed = PROXIMITY_SPEED_KMH[mode] || PROXIMITY_SPEED_KMH.walking;
  const minutes = proximity.minutes || PROXIMITY_DEFAULT_MINUTES[mode] || PROXIMITY_DEFAULT_MINUTES.walking;
  const km = (minutes / 60) * speed;
  return Math.min(Math.max(Math.round(km * 10) / 10, 0.3), 20);
}

function getDistanceKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return Math.round(R * c * 10) / 10;
}

// The tiered caps below exist so a search with only weak matches doesn't pad
// itself out with junk — that logic stays regardless of what the user asked
// for. `userMax` (the "Max Results" setting) only raises the ceiling for
// genuinely strong result sets; it never forces weak matches to be shown.
function getResultCap(results, userMax = 10) {
  if (results.length === 0) return 0;
  const topScore = Math.max(...results.map(r => r.score));
  if (topScore >= 90) return userMax;
  if (topScore >= 70) return Math.min(7, userMax);
  if (topScore >= 50) return Math.min(5, userMax);
  if (topScore >= 30) return Math.min(3, userMax);
  return Math.min(1, userMax);
}

function priceLevel(level) {
  const map = { 0: "Under 10 USD", 1: "Under 10 USD", 2: "10-30 USD", 3: "30-60 USD", 4: "60+ USD" };
  return map[level] || null;
}

// Extracts the raw text from a Claude API response, with clear diagnostics
// if the model didn't return a usable text block (e.g. truncation, refusal,
// or an unexpected content block type).
function extractText(response, label) {
  const blocks = response.content || [];

  if (blocks.length === 0) {
    console.error(`[${label}] response.content was empty:`, JSON.stringify(response, null, 2));
    throw new Error(`${label}: empty response from Claude`);
  }

  // Claude Sonnet 5 uses adaptive thinking by default, so the response can
  // include a "thinking" block before the actual "text" block. Find the
  // text block wherever it is, instead of assuming it's content[0].
  const textBlock = blocks.find(b => b.type === "text" && typeof b.text === "string");

  if (!textBlock) {
    console.error(`[${label}] No text block found. Block types were:`, blocks.map(b => b.type).join(", "));
    throw new Error(`${label}: no text block in response (got: ${blocks.map(b => b.type).join(", ")})`);
  }

  if (response.stop_reason === "max_tokens") {
    console.warn(`[${label}] WARNING: response was cut off (stop_reason: max_tokens). Consider raising max_tokens.`);
  }

  return textBlock.text;
}

async function fetchFromGoogle(query, analysis, userLat, userLng, radiusKm) {
  const lat = userLat || DEFAULT_LAT;
  const lng = userLng || DEFAULT_LNG;

  if (lat == null || lng == null) {
    throw new Error("No location available — browser geolocation failed and no default is set.");
  }

  const rawCuisine = analysis.cuisine || "";
  const cuisines = rawCuisine
    .split(/\s+or\s+/i)
    .map(c => c.trim())
    .filter(Boolean);

  let searchTerms;
  if (analysis.dish) {
    searchTerms = [analysis.dish];
  } else if (cuisines.length > 1) {
    searchTerms = cuisines;
  } else if (cuisines.length === 1) {
    searchTerms = cuisines;
  } else {
    searchTerms = [query];
  }

  const locationHint = analysis.location ? ` near ${analysis.location}` : "";

  // Don't append "restaurant" when the user is searching by an actual
  // restaurant name — chain or independent — it confuses Google Places.
  // Was previously inferred as "!cuisine && dish", which is also true for any
  // plain dish search (e.g. "pizza") — that stripped "restaurant" and the
  // type=restaurant filter from every dish-only query, degrading results
  // (occasionally down to zero in less dense areas). Trust the actual
  // classification instead of guessing from field nullness.
  const isBrandSearch = analysis.is_brand === true;

  // Each search term is an independent Google Places request — fire them in
  // parallel instead of awaiting one at a time (this only matters for
  // multi-cuisine queries like "Japanese or Italian", where searchTerms has
  // more than one entry).
  const termResults = await Promise.all(searchTerms.map(async (term) => {
    // No longer hardcodes "Osaka Japan" — relies on the `location` bias param
    // (lat/lng) below, which now reflects the user's real position.
    const googleQuery = isBrandSearch
      ? `${term}${locationHint}`
      : `${term} restaurant${locationHint}`;
    console.log(`Google Places query: "${googleQuery}" (near ${lat}, ${lng})`);

    const params = {
      query: googleQuery,
      location: `${lat},${lng}`,
      radius: Math.round(radiusKm * 1000),
      language: "en",
      key: GOOGLE_API_KEY,
    };

    if (!isBrandSearch) {
      params.type = "restaurant";
    }

    const response = await axios.get(
      "https://maps.googleapis.com/maps/api/place/textsearch/json",
      { params }
    );

    if (response.data.status !== "OK" && response.data.status !== "ZERO_RESULTS") {
      throw new Error(`Google Places error: ${response.data.status}`);
    }

    return { term, results: response.data.results || [] };
  }));

  // Dedupe by name first, keeping every candidate's computed distance —
  // the radius decision happens after, once we know how many candidates
  // actually fall within it.
  const candidatesByName = new Map();
  for (const { term, results } of termResults) {
    for (const place of results.slice(0, 20)) {
      if (candidatesByName.has(place.name)) continue;
      const placeLat = place.geometry?.location?.lat;
      const placeLng = place.geometry?.location?.lng;
      const distance = placeLat && placeLng ? getDistanceKm(lat, lng, placeLat, placeLng) : null;
      candidatesByName.set(place.name, { place, term, distance });
    }
  }
  const allCandidates = [...candidatesByName.values()];

  // Google's Text Search API treats radius/location as a ranking bias, not a
  // hard filter — it can return places well outside the requested radius if
  // they match the query text strongly (occasionally from a different city
  // entirely). Enforce the requested radius ourselves — but if the area is
  // genuinely sparse and nothing sits within it (e.g. the nearest pizza place
  // is 7 km out when the radius was 5), fall back to a looser cutoff instead
  // of returning zero results for what's actually just a real sparse area.
  let selected = allCandidates.filter(c => c.distance === null || c.distance <= radiusKm);
  if (selected.length === 0 && allCandidates.length > 0) {
    const looseRadiusKm = radiusKm * 3;
    selected = allCandidates.filter(c => c.distance === null || c.distance <= looseRadiusKm);
    console.log(`No places within ${radiusKm} km — falling back to ${looseRadiusKm} km (${selected.length} found)`);
  }

  const allPlaces = new Map();
  for (const { place, term, distance } of selected) {
    allPlaces.set(place.name, {
      name: place.name,
      place_id: place.place_id,
      photo_reference: place.photos?.[0]?.photo_reference || null,
      // Google doesn't label these by subject (food vs. interior vs.
      // storefront) — this is just whatever mix of photos exists for the
      // place, capped at 5. The client only fetches the extra ones (beyond
      // the hero shot) when a card is actually expanded, to keep Photo API
      // billing proportional to what's actually viewed.
      photos: (place.photos || []).slice(0, 5).map(p => p.photo_reference).filter(Boolean),
      cuisine: term,
      location: place.vicinity || "",
      price: priceLevel(place.price_level) || (place.rating ? `Rated ${place.rating}★` : "See Google Maps"),
      price_level: place.price_level ?? null,
      latitude: place.geometry?.location?.lat,
      longitude: place.geometry?.location?.lng,
      distance_km: distance,
      open_now: place.opening_hours?.open_now ?? null,
      rating: place.rating || null,
      review_count: place.user_ratings_total || 0,
      description: `${place.name} is located in ${place.vicinity || "the area"}. ${place.rating ? `Rated ${place.rating}/5 based on ${place.user_ratings_total} reviews.` : ""} ${place.opening_hours?.open_now !== undefined ? (place.opening_hours.open_now ? "Currently open." : "Currently closed.") : ""}`,
      reviews: [
        place.rating ? `Rated ${place.rating}/5 stars by ${place.user_ratings_total || 0} Google reviewers.` : "No rating available.",
        place.opening_hours?.open_now !== undefined ? (place.opening_hours.open_now ? "Currently open." : "Currently closed.") : "Opening hours unknown.",
        place.vicinity ? `Located at ${place.vicinity}.` : "",
        place.price_level !== undefined ? `Price level: ${priceLevel(place.price_level)}.` : "Price unknown.",
      ].filter(Boolean),
      // Filled in later, only for the shortlisted candidates that reach
      // scoring — real excerpts from Google reviewers, used as ground
      // truth for atmosphere/noise/dish claims instead of guesswork.
      customer_reviews: [],
    });
  }

  return [...allPlaces.values()];
}

// Text Search never returns review text, only aggregate rating/count — so
// claims about atmosphere, noise, or specific dishes had no real evidence
// behind them. Place Details (billed separately, "Atmosphere" data) returns
// up to 5 real review excerpts per place, fetched only for the shortlisted
// candidates that make it to scoring.
// One Place Details call per shortlisted restaurant already happens for
// review text — pull phone/website in the same request (Contact-category
// fields) instead of a second billed call, so "Call"/"Website" buttons don't
// double the Place Details cost per search.
// Reviews/phone/website don't change minute to minute, but a multi-turn
// refinement session ("quiet ramen" → "cheap" → "no seafood") re-shortlists
// mostly the same restaurants every turn — without this, each turn re-fetches
// (and re-bills) Place Details for restaurants we already just looked up.
// In-memory only: resets on restart, not shared across processes — fine for
// a single-process POC; a real deployment would want Redis or similar.
const PLACE_DETAILS_CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes
const placeDetailsCache = new Map(); // place_id -> { data, expiresAt }

// Text Search's opening_hours.open_now is just a boolean — Place Details'
// "periods" (day 0-6 Sun-Sat, "HHMM" local time at the place) has the actual
// schedule, which lets us say *when* it opens/closes instead of just
// open-or-not. Assumes the server's local time is a reasonable stand-in for
// the restaurant's own — true for a single-city POC, would need the place's
// real UTC offset (Google doesn't return one) for a multi-timezone app.
function formatClockTime(hhmm) {
  const hour = parseInt(hhmm.slice(0, 2), 10);
  const minute = hhmm.slice(2);
  const period = hour >= 12 ? "PM" : "AM";
  const hour12 = hour % 12 || 12;
  return minute === "00" ? `${hour12} ${period}` : `${hour12}:${minute} ${period}`;
}

function formatHoursStatus(periods) {
  if (!periods || periods.length === 0) return null;

  const alwaysOpen = periods.length === 1 && periods[0].open?.time === "0000" && !periods[0].close;
  if (alwaysOpen) return "Open 24 hours";

  const now = new Date();
  const today = now.getDay();
  const nowMinutes = now.getHours() * 60 + now.getMinutes();
  const toMinutes = (t) => parseInt(t.slice(0, 2), 10) * 60 + parseInt(t.slice(2), 10);

  for (const p of periods) {
    if (!p.open || p.open.day !== today || !p.close) continue;
    const openMin = toMinutes(p.open.time);
    let closeMin = toMinutes(p.close.time);
    if (p.close.day !== p.open.day) closeMin += 24 * 60; // closes after midnight
    if (nowMinutes >= openMin && nowMinutes < closeMin) {
      return `Closes ${formatClockTime(p.close.time)}`;
    }
  }

  const dayNames = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
  for (let offset = 0; offset < 7; offset++) {
    const checkDay = (today + offset) % 7;
    const upcoming = periods
      .filter(p => p.open && p.open.day === checkDay && (offset > 0 || toMinutes(p.open.time) > nowMinutes))
      .sort((a, b) => toMinutes(a.open.time) - toMinutes(b.open.time));
    if (upcoming.length === 0) continue;
    const label = formatClockTime(upcoming[0].open.time);
    if (offset === 0) return `Opens ${label}`;
    if (offset === 1) return `Opens ${label} tomorrow`;
    return `Opens ${label} ${dayNames[checkDay]}`;
  }
  return null;
}

async function fetchPlaceDetails(placeId) {
  const empty = { reviews: [], phone: null, website: null, hours_status: null, photos: [] };
  if (!placeId) return empty;

  const cached = placeDetailsCache.get(placeId);
  if (cached && cached.expiresAt > Date.now()) {
    console.log(`Place Details cache hit: ${placeId}`);
    return cached.data;
  }

  try {
    const response = await axios.get(
      "https://maps.googleapis.com/maps/api/place/details/json",
      // Text Search only ever returns one representative photo per place —
      // Place Details' own "photo" field returns up to 10, so the gallery
      // pulls from here instead of the thin Text Search list.
      { params: { place_id: placeId, fields: "review,formatted_phone_number,website,opening_hours,photo", key: GOOGLE_API_KEY } }
    );
    if (response.data.status !== "OK") return empty;
    const result = response.data.result || {};
    const data = {
      reviews: (result.reviews || []).map(r => r.text).filter(Boolean),
      phone: result.formatted_phone_number || null,
      website: result.website || null,
      hours_status: formatHoursStatus(result.opening_hours?.periods),
      photos: (result.photos || []).slice(0, 5).map(p => p.photo_reference).filter(Boolean),
    };
    placeDetailsCache.set(placeId, { data, expiresAt: Date.now() + PLACE_DETAILS_CACHE_TTL_MS });
    return data;
  } catch (error) {
    console.error(`Place Details failed for ${placeId}:`, error.message);
    return empty;
  }
}

async function analyzeQuery(userQuery, previousQuery) {
  const context = previousQuery
    ? `The user previously searched: "${previousQuery}"\nNow they are refining with: "${userQuery}"`
    : `Query: "${userQuery}"`;

  const response = await client.messages.create({
    model: "claude-haiku-4-5", // structured field extraction, not nuanced judgment — Haiku is
                                // plenty, and (unlike Sonnet 5) runs with no thinking by default,
                                // so there's no adaptive-reasoning overhead to disable here.
    max_tokens: 600,
    messages: [{
      role: "user",
      content: `You are a restaurant search assistant. Analyze this search query and extract what the user is looking for.

${context}

Pay special attention to negative requirements — things the user does NOT want. Words like "not", "no", "without", "avoid", "but not", "nothing too" indicate a hard requirement to exclude.

${previousQuery ? `Also detect the user's INTENT based on their new message:
- "refine": they are adjusting or adding to their previous search (signals: "but", "also", "more", "less", "with", "without", "cheaper", "closer", "quieter", "and", adding a constraint)
- "pivot": they want a completely different cuisine or concept (signals: "forget that", "never mind", "something else", "completely different", "no wait"). Note: "actually I want X" should only pivot the cuisine/concept, NOT discard constraints like price or audience unless explicitly said.
- "new": they are explicitly starting fresh (signals: "new search", "start over", "reset", "from scratch")
If the message is ambiguous, default to "refine".` : ""}

CUISINE RULES:
- If the user specifies a specific dish (e.g. "wagyu", "ramen", "carbonara"), set "dish" and set "cuisine" to null. Do NOT keep a generic cuisine tag when a specific dish has been named. Set "is_brand" to false — a dish name is not a brand.
- If the user says "Japanese or Italian", set cuisine to "Japanese or Italian" exactly.
- Never infer a sub-cuisine (e.g. "sushi") from a broad term like "Japanese" — keep it broad.
- If the user names a specific restaurant — whether a chain/brand (e.g. "Starbucks", "McDonald's", "Ippudo", "Ichiran") or a single independent place by its actual name (e.g. "find Pizzarella", "is Coq en Pate open") — set "dish" to that exact name, set "cuisine" to null, and set "is_brand" to true. This is the ONLY case where "is_brand" should be true — it controls whether "restaurant" gets appended to the Google Places query (appending it to an actual name, e.g. "Starbucks restaurant" or "Pizzarella restaurant", confuses the search) and whether scoring is by name match instead of cuisine/vibe fit. Do NOT set it for a generic craving like "pizza" or "a cozy Italian place" — only an actual proper name the user is searching for by that name.
- When "is_brand" is true, naming a different specific restaurant is never a continuation of whatever the previous search was about — it's its own standalone lookup, even if "intent" comes out as "refine" or "pivot" for other reasons. Keep "interpretation" to something like "Search for a restaurant called X" — do NOT describe it as refining, pivoting from, or otherwise related to the previous search.

ATMOSPHERE RULES:
- "quiet", "relaxed", "peaceful", "chill", "unwind", "low-key" mean LOW NOISE and CASUAL PACE — NOT upscale, NOT omakase, NOT formal. Only use "upscale" or "fine dining" atmosphere if the user explicitly says those words.

PROXIMITY RULES:
- Detect how close the user wants results and by what mode of travel, if stated.
- "near me", "close by", "nearby" with no travel mode mentioned → mode: null, minutes: null.
- "walking distance", "short walk", "walkable" → mode: "walking", minutes: null (or the stated number, e.g. "10 minute walk" → minutes: 10).
- "bike distance", "biking distance", "bike ride" → mode: "biking", minutes: null or the stated number.
- "X minutes/mins away" with no mode stated → mode: null, minutes: X.
- "X minute drive", "driving distance" → mode: "driving", minutes: X or null.
- If the query says nothing at all about proximity or travel time, set "proximity" to null entirely (not an object).

DIRECTIONS RULE:
- Set "wants_directions" to true ONLY if the user is explicitly asking to navigate/route to a place — "directions to X", "how do I get to X", "navigate to X", "take me to X", "route to X". This is about wanting to go there right now, not about researching or discovering options.
- A plain "find X" or "is X open" is NOT a directions request — set it false.

Respond ONLY with a valid JSON object. No comments, no extra text, no markdown:
{
  "cuisine": "specific cuisine, 'Japanese or Italian', or null",
  "dish": "specific dish or null",
  "is_brand": true or false — true ONLY if "dish" is an actual restaurant name the user searched for by name, chain or independent (e.g. Starbucks, Ippudo, Pizzarella), false for a generic dish (e.g. pizza, ramen) or when dish is null,
  "wants_directions": true or false — true only per the DIRECTIONS RULE above,
  "atmosphere": "vibe or null",
  "occasion": "occasion or null",
  "audience": "who they are dining with or null",
  "price": "budget as a plain phrase or null",
  "location": "specific sub-area or landmark within the city (e.g. Dotonbori, Shinsaibashi, Umeda) — NOT the city name itself. Null if no specific area mentioned.",  "priority": "the single most important thing in this query",
  "must_not": ["things explicitly NOT wanted — empty array if none"],
  "time_sensitive": true or false — true if query implies immediacy (tonight, now, hungry, for dinner, want to go). false for research/planning queries,
  "proximity": {"mode": "walking, biking, driving, or null", "minutes": "number or null"} or null if no proximity/travel-time cue at all,
"interpretation": "One sentence describing what the user wants right now, as if speaking directly about their goal. Never mention 'pivot', 'refine', 'previous search', or any meta language. Just describe the desired outcome.",  "intent": ${previousQuery ? `"refine", "pivot", or "new"` : `"new"`}
}`,
    }],
  });

  const raw = extractText(response, "analyzeQuery").replace(/```json|```/g, "").trim();
  const start = raw.indexOf('{');
  const end = raw.lastIndexOf('}');
  if (start === -1 || end === -1) throw new Error('No JSON object found in analyzeQuery response');
  return JSON.parse(raw.slice(start, end + 1));
}

// Used only after a tag removal (see the `directAnalysis` path in /search) —
// the fields are already known, so this just re-phrases the "What I
// understood" sentence to match them, without paying for a full
// re-classification call.
async function regenerateInterpretation(analysis) {
  const response = await client.messages.create({
    model: "claude-haiku-4-5",
    max_tokens: 100,
    messages: [{
      role: "user",
      content: `Write ONE sentence describing what this restaurant search is looking for right now, based on these fields. Speak directly about the goal, as if describing what the user wants. Never mention field names, "null", JSON, or meta language like "refine"/"pivot"/"intent". Silently skip any field that's null.

${JSON.stringify(analysis, null, 2)}

Respond with ONLY the sentence — no quotes, no markdown, no preamble.`,
    }],
  });
  return extractText(response, "regenerateInterpretation").trim();
}

const MERGE_FIELDS = ["atmosphere", "occasion", "audience", "price", "location", "proximity"];

// Combines this turn's freshly-extracted fields with the previous turn's
// merged state. A field only overrides the prior value when the user
// actually mentioned it this turn (non-null) — otherwise earlier constraints
// like price or audience survive a pivot, matching what the intent-
// classification prompt already instructs ("actually I want X" should only
// pivot cuisine/concept, not silently drop everything else the user said).
function mergeAnalysis(previousAnalysis, newAnalysis) {
  if (!previousAnalysis || newAnalysis.intent === "new") {
    return newAnalysis;
  }

  const merged = { ...previousAnalysis };

  // Cuisine and dish are mutually exclusive — whichever one was named this
  // turn wins and clears the other; if neither was mentioned, keep whatever
  // was already set.
  if (newAnalysis.cuisine !== null) {
    merged.cuisine = newAnalysis.cuisine;
    merged.dish = null;
    merged.is_brand = false; // a cuisine is never a brand
  } else if (newAnalysis.dish !== null) {
    merged.dish = newAnalysis.dish;
    merged.cuisine = null;
    merged.is_brand = newAnalysis.is_brand === true;
  }

  for (const field of MERGE_FIELDS) {
    if (newAnalysis[field] !== null && newAnalysis[field] !== undefined) {
      merged[field] = newAnalysis[field];
    }
  }

  // Exclusions accumulate across turns instead of being replaced.
  const prevMustNot = previousAnalysis.must_not || [];
  const newMustNot = newAnalysis.must_not || [];
  merged.must_not = [...new Set([...prevMustNot, ...newMustNot])];

  // Always reflects the latest turn — "directions to X" this turn shouldn't
  // keep auto-jumping to the map on every later refinement.
  merged.priority = newAnalysis.priority;
  merged.time_sensitive = newAnalysis.time_sensitive;
  merged.interpretation = newAnalysis.interpretation;
  merged.intent = newAnalysis.intent;
  merged.wants_directions = newAnalysis.wants_directions === true;

  return merged;
}

// "Cheap"/"expensive"/"moderate" are judged relative to what's actually
// nearby, not a fixed global dollar tier — Copenhagen's cheapest ramen and
// Hanoi's cheapest ramen carry wildly different Google price_level values,
// but "cheap" should mean the same thing (the bottom of what's locally
// available) in both places.
function filterByRelativePrice(restaurantData, rank) {
  if (rank === "none") return restaurantData;

  const levels = [...new Set(
    restaurantData.map(r => r.price_level).filter(l => l !== null && l !== undefined)
  )].sort((a, b) => a - b);

  if (levels.length === 0) return restaurantData; // no price data at all to be relative about

  const min = levels[0];
  const max = levels[levels.length - 1];

  if (rank === "low") {
    return restaurantData.filter(r => r.price_level === null || r.price_level === min);
  }
  if (rank === "high") {
    return restaurantData.filter(r => r.price_level === null || r.price_level === max);
  }
  // "mid": exclude both extremes — but only when there's an actual spread to
  // exclude from; with just 1-2 distinct levels present there's no meaningful
  // "middle" to isolate.
  if (levels.length < 3) return restaurantData;
  return restaurantData.filter(r => r.price_level === null || (r.price_level !== min && r.price_level !== max));
}

function applyPriceFilter(restaurantData, priceStr) {
  const relativeRank = parseRelativePriceRank(priceStr);
  if (relativeRank) {
    const beforeCount = restaurantData.length;
    const filtered = filterByRelativePrice(restaurantData, relativeRank);
    console.log(`Relative price filter ("${relativeRank}" of what's nearby): ${beforeCount} → ${filtered.length} restaurants`);
    return filtered;
  }

  const priceCeiling = parsePriceCeiling(priceStr);
  if (priceCeiling !== null && priceCeiling < 4) {
    const beforeCount = restaurantData.length;
    const strict = restaurantData.filter(r =>
      r.price_level === null || r.price_level <= priceCeiling
    );

    // Google's price_level tiers don't account for regional cost of living —
    // an entire city's worth of results can sit one tier above the ceiling
    // (e.g. every option tagged "Moderate" when the user asked for "under
    // $15"), which would otherwise wipe out every candidate before scoring
    // even runs. Fall back to a one-tier-looser cutoff before giving up on
    // price filtering altogether.
    let filtered = strict;
    if (filtered.length === 0) {
      const loose = restaurantData.filter(r => r.price_level === null || r.price_level <= priceCeiling + 1);
      filtered = loose.length > 0 ? loose : restaurantData;
    }
    console.log(`Price filter (≤ level ${priceCeiling}): ${beforeCount} → ${filtered.length} restaurants`);
    return filtered;
  }

  return restaurantData;
}

async function searchRestaurants(userQuery, analysis, restaurantData, radiusKm, maxResults) {
  let filteredData = applyPriceFilter(restaurantData, analysis.price);

  console.log(`Fetching real reviews for ${filteredData.length} shortlisted restaurants...`);
  filteredData = await Promise.all(filteredData.map(async (r) => {
    const details = await fetchPlaceDetails(r.place_id);
    // Prefer Place Details' fuller photo list over Text Search's single one
    // — fall back to whatever Text Search gave if Details somehow returned
    // none at all.
    const photos = details.photos.length > 0 ? details.photos : r.photos;
    // Google's photo_reference tokens aren't stable across different API
    // calls — the SAME underlying photo gets a different reference string
    // from Text Search vs. Place Details, so comparing the hero's Text-
    // Search-derived reference against Place Details' list (client-side,
    // to skip the duplicate in the gallery) silently never matched. Once
    // Place Details' own list is available, make it the one source of
    // truth for both the hero and the gallery so they share a reference
    // universe and the client's dedup-by-string-match actually works.
    const photo_reference = photos.length > 0 ? photos[0] : r.photo_reference;
    return { ...r, customer_reviews: details.reviews, phone: details.phone, website: details.website, hours_status: details.hours_status, photos, photo_reference };
  }));

  const response = await client.messages.create({
    model: "claude-sonnet-5",
    max_tokens: 8192, // raised from 4096 — 4096 still truncated mid-response for
                       // broad queries where most/all 20 restaurants score above 30
                       // and each gets a full summary/evidence/tags object.
    thinking: { type: "disabled" }, // scoring follows an explicit rubric, not open-ended
                                     // reasoning — adaptive thinking here was pure latency.
                                     // Revisit (drop this, or use output_config.effort:"low"
                                     // instead) if scoring quality regresses.
    messages: [{
      role: "user",
      content: `You are a restaurant discovery AI.

The user searched: "${userQuery}"

Here is your analysis of what they want:
${JSON.stringify(analysis, null, 2)}

Here is the restaurant database:
${JSON.stringify(filteredData, null, 2)}

Each restaurant's "customer_reviews" field (when non-empty) contains real excerpts
written by Google reviewers — this is your only real evidence for atmosphere, noise
level, and specific dishes. "reviews" and "description" are generated from
rating/hours/price metadata, not real customer text.

SCORING RULES:
- Score each restaurant from 0 to 100 based on how well it matches.
- For "must_not" and atmosphere judgments (e.g. "quiet", "not loud", "no seafood"): base the verdict on "customer_reviews" text when available. If a restaurant has no customer_reviews to confirm or deny a must_not claim, don't guess — treat it as unconfirmed rather than disqualifying it, and lower "confidence" to reflect that.
- CUISINE IS THE HIGHEST PRIORITY. Wrong cuisine = max score 35. Right cuisine = base score 60.
- If the user said "Japanese or Italian", both cuisines are equally valid — return a balanced mix, do NOT favour one over the other.
- If a specific dish was requested (e.g. wagyu, ramen), heavily weight restaurants likely to serve that dish.
- If "is_brand" is true (the user searched for a specific restaurant by name — a chain like Starbucks or an independent place like "Pizzarella"), the ONLY criteria is whether the restaurant name matches. Score any matching location 90+. Ignore cuisine scoring entirely for name searches.
- Factor in distance_km: under ${radiusKm} km = great, under ${Math.round(radiusKm * 2 * 10) / 10} km = good, under ${Math.round(radiusKm * 3 * 10) / 10} km = acceptable.
- If a specific location or landmark was requested, prioritise restaurants closest to it.
- "must_not" items are HARD DISQUALIFIERS — score below 20 if matched.
- ATMOSPHERE: "quiet", "relaxed", "peaceful", "chill", "unwind", "low-key" mean LOW NOISE and CASUAL PACE. If atmosphere contains any of these words, HARD PENALISE omakase, counter dining, fine dining, and formal restaurants — score them below 40 regardless of rating. A neighbourhood izakaya or casual ramen shop should outscore a Michelin-starred counter.
- OCCASION & AUDIENCE: "occasion" (e.g. "date night", "birthday", "business lunch") and "audience" (e.g. "with kids", "with grandma", "with colleagues", "solo") describe who's going and why. Check "customer_reviews" for direct evidence — reviewers mentioning kid-friendly, romantic, good for groups, wheelchair accessible, quiet booths, etc. If reviews confirm a good fit, factor it in as a positive. If reviews actively contradict it (e.g. audience is "with kids" but reviews describe a loud bar-only, 21+ scene; or occasion is "date night" but reviews describe a rowdy sports-bar vibe), apply a moderate penalty — not a hard disqualifier like "must_not". If there's no review evidence either way, treat it as neutral and don't guess.
- Use Google rating and review count as quality signals but do not let high ratings override a wrong vibe.
- Price has already been pre-filtered — do not penalise any restaurant in this list for price.
- Only return restaurants that genuinely match. Return [] if nothing scores above 30.
- Return at most ${maxResults} restaurants — the highest-scoring ones — even if more qualify.
- Distance is already shown to the user as its own tag — do NOT mention distance, km, or "away" in "summary" or "evidence". Use that space to talk about the restaurant itself: cuisine, dish, atmosphere, rating, reviews, why it fits.

Respond ONLY with a JSON array, no other text:
[
  {
    "name": "Restaurant Name",
    "score": 92,
    "summary": "Why this matched (1-2 sentences, specific, no distance/km mentions)",
    "evidence": ["evidence from description or reviews, no distance/km mentions"],
    "confidence": "high, medium, or low",
    "tags": {
      "cuisine": "Italian",
      "price": "10-30 USD",
      "distance": "0.5 km",
      "vibe": "cozy"
    }
  }
]`,
    }],
  });

  const raw = extractText(response, "searchRestaurants").replace(/```json|```/g, "").trim();
  const start = raw.indexOf('[');
  const end = raw.lastIndexOf(']');
  if (start === -1 || end === -1) {
    console.error("searchRestaurants: could not find a JSON array in the response. Raw text was:\n", raw);
    throw new Error('No JSON array found in searchRestaurants response');
  }
  const scored = JSON.parse(raw.slice(start, end + 1));

  // Claude's scoring response only carries name/score/summary/etc — phone,
  // website, Google's own rating, place_id, hours, and the Place-Details
  // photo list all live on filteredData (from the Place Details enrichment
  // above / the original Places fields) and need to be reattached so the
  // client can render Call/Website buttons, show the real rating alongside
  // our own match score, ask follow-up questions (place_id), and show extra
  // photos beyond the hero shot.
  return scored.map(r => {
    const match = filteredData.find(d => d.name === r.name);
    return match ? { ...r, phone: match.phone, website: match.website, rating: match.rating, review_count: match.review_count, place_id: match.place_id, hours_status: match.hours_status, photos: match.photos, photo_reference: match.photo_reference } : r;
  });
}

// Answers a free-form question about one already-found restaurant, grounded
// only in what we actually know about it (the same real review excerpts
// used for scoring, plus the summary/tags already shown) — Haiku is fast
// enough here that this doesn't need the staged-progress treatment /search
// gets, and there's nothing to classify, so no analyzeQuery step.
async function askAboutRestaurant(name, reviews, tags, summary, question) {
  const reviewText = reviews && reviews.length > 0
    ? reviews.map(r => `"${r}"`).join("\n")
    : "No customer reviews available.";

  const response = await client.messages.create({
    model: "claude-haiku-4-5",
    max_tokens: 300,
    messages: [{
      role: "user",
      content: `Answer a question about a specific restaurant for someone deciding whether to go. Only use the information given below — never invent hours, menu items, prices, or reviews that aren't there. If the answer isn't in the given information, say that plainly instead of guessing.

Restaurant: ${name}
What we know: ${summary || "nothing beyond the name"}
Tags: ${JSON.stringify(tags || {})}
Customer reviews:
${reviewText}

Question: ${question}

Answer in 1-3 short, direct, conversational sentences.`,
    }],
  });

  return extractText(response, "askAboutRestaurant").trim();
}

// Gives a browse-mode marker the same kind of description a real search's
// "summary" field carries — same model, same grounded-only-in-real-data
// rule, same 1-2 sentence length — but written as a general description of
// the place rather than "why this matched", since there's no query behind
// a plain nearby-browse tap. Only called per marker someone actually taps
// (see /place-details), never for a whole nearby list at once.
async function describeRestaurant(name, reviews, rating, reviewCount) {
  const reviewText = reviews && reviews.length > 0
    ? reviews.slice(0, 5).map(r => `"${r}"`).join("\n")
    : "No customer reviews available.";

  const response = await client.messages.create({
    model: "claude-haiku-4-5",
    max_tokens: 60, // hard ceiling on top of the prompt's own length rule — a
                     // 150-token budget let earlier attempts run to 5+ sentences
                     // despite being told "1-2"; this makes long output impossible.
    messages: [{
      role: "user",
      content: `Write a ONE-sentence, appealing description of this restaurant, in the same style as a short curated recommendation blurb — like "Highest-rated French option with an outstanding 4.8 rating, praised for authentic, fresh food and a cozy, welcoming atmosphere." Maximum 25 words. Ground it only in the rating and reviews given below — never invent cuisine, dishes, or atmosphere details that aren't supported by them.

Restaurant: ${name}
Google rating: ${rating ?? "unknown"} (${reviewCount ?? 0} reviews)
Customer reviews:
${reviewText}

Respond with ONLY that one sentence — no preamble, no quotation marks around it.`,
    }],
  });

  return extractText(response, "describeRestaurant").trim();
}

app.post("/ask-restaurant", async (req, res) => {
  const { placeId, name, question, tags, summary } = req.body;
  if (!placeId || !name || !question) {
    return res.status(400).json({ error: "placeId, name, and question are required" });
  }

  try {
    const details = await fetchPlaceDetails(placeId);
    const answer = await askAboutRestaurant(name, details.reviews, tags, summary, question);
    res.json({ answer });
  } catch (error) {
    console.error("Ask-restaurant error:", error.message);
    res.status(500).json({ error: error.message });
  }
});

app.post("/search", async (req, res) => {
  const userQuery = req.body.query || null;
  const previousQuery = req.body.previousQuery || null;
  const previousAnalysis = req.body.previousAnalysis || null;
  // Set when the user removes a constraint tag client-side instead of typing
  // a new query — skips re-classification and uses the edited analysis as-is.
  const directAnalysis = req.body.directAnalysis || null;
  const userLat = req.body.latitude || null;
  const userLon = req.body.longitude || null;
  // Clamp to a sane range regardless of what the client sends — this only
  // raises the ceiling for genuinely strong result sets (see getResultCap).
  const maxResults = Math.min(Math.max(parseInt(req.body.maxResults, 10) || 5, 1), 20);

  console.log("Search received:", userQuery || "(tag removed, no new query)");

  try {
    let analysis;
    let isPivotOrNew;

    if (directAnalysis) {
      // Tag removal / rerunning a saved history entry replay the last
      // analysis as-is — neither is itself a directions request, so don't
      // let a stale true from whatever search this analysis came from
      // silently re-trigger the map auto-jump.
      analysis = { ...directAnalysis, intent: "refine", wants_directions: false };
      analysis.interpretation = await regenerateInterpretation(analysis);
      isPivotOrNew = false;
      console.log("Using edited analysis (tag removed):", JSON.stringify(analysis));
    } else {
      console.log("Calling analyzeQuery...");
      const newAnalysis = await analyzeQuery(userQuery, previousQuery);
      console.log("Analysis done:", JSON.stringify(newAnalysis));

      analysis = mergeAnalysis(previousAnalysis, newAnalysis);
      isPivotOrNew = newAnalysis.intent === "pivot" || newAnalysis.intent === "new";
      console.log("Merged analysis:", JSON.stringify(analysis));
    }

    // Structured `analysis` now carries the full accumulated state, so each
    // turn's search text only needs to be this turn's own message — no more
    // concatenating the whole conversation into one run-on string.
    const fullQuery = userQuery || analysis.interpretation || analysis.priority || "restaurants";

    const radiusKm = computeRadiusKm(analysis.proximity);
    console.log(`Proximity: ${JSON.stringify(analysis.proximity)} → radius ${radiusKm} km`);

    console.log("Fetching from Google Places...");
    const restaurantData = await fetchFromGoogle(fullQuery, analysis, userLat, userLon, radiusKm);
    console.log(`Got ${restaurantData.length} restaurants from Google`);

    console.log("Calling searchRestaurants...");
    const results = await searchRestaurants(fullQuery, analysis, restaurantData, radiusKm, maxResults);

    results.forEach(r => {
      const match = restaurantData.find(d => d.name === r.name);
      if (match) {
        r.latitude = match.latitude;
        r.longitude = match.longitude;
        r.distance_km = match.distance_km;
        r.open_now = match.open_now;
        // photo_reference and photos are NOT reattached here — searchRestaurants
        // already set both from the same Place Details call (the fuller photo
        // set, up to 10, vs. Text Search's single one on restaurantData), so
        // they share one reference-string universe and the client's dedup
        // between hero and gallery actually matches. Overwriting photo_reference
        // from restaurantData here would reintroduce a hero/gallery duplicate,
        // since Text Search and Place Details tokens differ for the same photo.
      }
    });

    // Enforce 60% max per cuisine in multi-cuisine searches
    const rawCuisine = analysis.cuisine || "";
    const isMultiCuisine = rawCuisine.toLowerCase().includes(' or ');
    if (isMultiCuisine && results.length > 2) {
      const cuisineCounts = {};
      const maxPerCuisine = Math.ceil(results.length * 0.6);
      const balanced = [];
      const sorted = [...results].sort((a, b) => b.score - a.score);
      for (const r of sorted) {
        const c = (r.tags?.cuisine || 'unknown').toLowerCase();
        cuisineCounts[c] = (cuisineCounts[c] || 0) + 1;
        if (cuisineCounts[c] <= maxPerCuisine) balanced.push(r);
      }
      results.length = 0;
      results.push(...balanced);
    }
    console.log("Results done:", results.length, "results");

    // For brand searches, accept any result if Google returned it
    const isBrandQuery = analysis.is_brand === true;
    const finalResults = isBrandQuery
      ? results.filter(r => r.score > 0)
      : results;
    const timeSensitive = analysis.time_sensitive;
    const sortedResults = finalResults.sort((a, b) => {
      if (timeSensitive) {
        const aOpen = a.open_now === true ? 1 : 0;
        const bOpen = b.open_now === true ? 1 : 0;
        if (bOpen !== aOpen) return bOpen - aOpen;
      }
      return b.score - a.score;
    });
    const cap = getResultCap(sortedResults, maxResults);
    const cappedResults = sortedResults.slice(0, cap);

    res.json({ analysis, results: cappedResults, intent: analysis.intent, clearedContext: isPivotOrNew });

  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: error.message || "Something went wrong" });
  }
});

// Lets the user override automatic geolocation by typing an area instead —
// needed when the device's own location is wrong or too imprecise to trust
// (e.g. a laptop with no GPS/WiFi-positioning signal falling back to ~200km-
// accurate IP estimation), and separately useful for planning a search in a
// city the user isn't currently in.
app.get("/geocode", async (req, res) => {
  const address = req.query.address;
  if (!address) return res.status(400).json({ error: "Missing address" });

  try {
    const response = await axios.get(
      "https://maps.googleapis.com/maps/api/geocode/json",
      { params: { address, key: GOOGLE_API_KEY } }
    );
    if (response.data.status !== "OK" || !response.data.results?.length) {
      return res.status(404).json({ error: `Couldn't find "${address}"` });
    }
    const result = response.data.results[0];
    res.json({
      lat: result.geometry.location.lat,
      lng: result.geometry.location.lng,
      formatted_address: result.formatted_address,
    });
  } catch (error) {
    console.error("Geocode failed:", error.message);
    res.status(502).json({ error: "Geocoding failed" });
  }
});

// Start-screen showcase — a real nearby restaurant per category, to give a
// sense of both "you can search for X" and "here's what you get back
// (rating, photo)" before anyone's actually searched. Deliberately cheap:
// no Claude call (the categories are fixed, nothing to classify) and no
// Place Details call (name/rating/price/photo are already free in the Text
// Search response) — just one Text Search request per category, picking
// the highest-rated result. Not an AI-vetted "best match," just a real one.
const SHOWCASE_CATEGORIES = [
  { label: "Cheap eats", query: "cheap eats near me", searchTerm: "cheap restaurant" },
  { label: "Family-friendly", query: "family-friendly restaurant nearby", searchTerm: "family friendly restaurant" },
  { label: "Cozy cafe", query: "cozy cafe nearby", searchTerm: "cafe" },
];

app.get("/showcase", async (req, res) => {
  const { lat, lng } = req.query;
  if (!lat || !lng) return res.status(400).json({ error: "Missing lat/lng" });

  try {
    const items = await Promise.all(SHOWCASE_CATEGORIES.map(async (cat) => {
      const response = await axios.get(
        "https://maps.googleapis.com/maps/api/place/textsearch/json",
        { params: { query: cat.searchTerm, location: `${lat},${lng}`, radius: 5000, key: GOOGLE_API_KEY } }
      );
      const candidates = (response.data.results || []).filter(p => p.rating);
      if (candidates.length === 0) return null;
      const best = candidates.sort((a, b) => b.rating - a.rating)[0];
      return {
        label: cat.label,
        query: cat.query,
        name: best.name,
        rating: best.rating,
        review_count: best.user_ratings_total || 0,
        photo_reference: best.photos?.[0]?.photo_reference || null,
      };
    }));
    res.json({ items: items.filter(Boolean) });
  } catch (error) {
    console.error("Showcase failed:", error.message);
    res.status(502).json({ error: "Showcase failed" });
  }
});

// Powers "browse the map" — a plain, un-scored list of what's actually
// nearby for someone who wants to look around rather than describe what
// they want. Same cost-conscious shape as /showcase: one Text Search call,
// no Claude, no per-place Details — just what Google already gives back.
app.get("/nearby", async (req, res) => {
  const { lat, lng } = req.query;
  if (!lat || !lng) return res.status(400).json({ error: "Missing lat/lng" });

  try {
    const response = await axios.get(
      "https://maps.googleapis.com/maps/api/place/textsearch/json",
      { params: { query: "restaurant", location: `${lat},${lng}`, radius: 2000, type: "restaurant", key: GOOGLE_API_KEY } }
    );
    const results = (response.data.results || [])
      .filter(p => p.geometry?.location)
      .map(p => ({
        name: p.name,
        rating: p.rating || null,
        review_count: p.user_ratings_total || 0,
        photo_reference: p.photos?.[0]?.photo_reference || null,
        place_id: p.place_id,
        latitude: p.geometry.location.lat,
        longitude: p.geometry.location.lng,
        distance_km: getDistanceKm(Number(lat), Number(lng), p.geometry.location.lat, p.geometry.location.lng),
      }))
      .sort((a, b) => a.distance_km - b.distance_km)
      .slice(0, 20);
    res.json({ results });
  } catch (error) {
    console.error("Nearby failed:", error.message);
    res.status(502).json({ error: "Nearby search failed" });
  }
});

// Lets a browse-mode marker fetch the same hours/phone/website a real
// search's Place Details enrichment already gets — lazily, one call per
// marker someone actually taps, not upfront for every nearby result. Split
// out from the description call (below) so the popup can show hours/phone/
// website as soon as this — the fast one, a single Google round trip —
// resolves, instead of both waiting on the slower LLM call together.
app.get("/place-details", async (req, res) => {
  const { place_id } = req.query;
  if (!place_id) return res.status(400).json({ error: "Missing place_id" });
  const details = await fetchPlaceDetails(place_id);
  res.json(details);
});

// The slower half of a browse-mode marker tap — takes the reviews the
// client already fetched via /place-details (no need to hit Google again)
// and writes the one-sentence description. Kept as its own request so the
// client can show hours/phone/website immediately and let this arrive
// separately, rather than blocking everything on an LLM call.
app.post("/describe-restaurant", async (req, res) => {
  const { name, rating, review_count, reviews } = req.body;
  if (!name) return res.status(400).json({ error: "Missing name" });
  let description = null;
  try {
    description = await describeRestaurant(name, reviews || [], rating ?? null, review_count ?? null);
  } catch (error) {
    console.error("describeRestaurant failed:", error.message);
  }
  res.json({ description });
});

// Proxies Google's Place Photo endpoint so the browser never sees our
// Google API key — the key has to go in that URL's querystring, so calling
// it directly from the client would leak it, breaking the pattern every
// other Google/Anthropic call in this app already follows (server-side only).
app.get("/photo", async (req, res) => {
  const ref = req.query.ref;
  if (!ref) return res.status(400).send("Missing ref");

  try {
    const response = await axios.get(
      "https://maps.googleapis.com/maps/api/place/photo",
      {
        params: { maxwidth: 400, photo_reference: ref, key: GOOGLE_API_KEY },
        responseType: "stream",
      }
    );
    res.set("Content-Type", response.headers["content-type"]);
    res.set("Cache-Control", "public, max-age=86400"); // a given photo_reference's image doesn't change
    response.data.pipe(res);
  } catch (error) {
    console.error("Photo proxy failed:", error.message);
    res.status(502).send("Photo unavailable");
  }
});

app.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});