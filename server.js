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

const PRICE_BUDGET_MAP = {
  "under 10": 1, "under $10": 1, "budget": 1, "cheap": 1,
  "10-15": 1, "$10-15": 1, "10-20": 1, "$10-20": 1, "around $10": 1, "around $15": 1,
  "10-30": 2, "$10-30": 2, "moderate": 2, "mid": 2, "around $20": 2, "around $25": 2, "around $30": 2,
  "30-60": 3, "$30-60": 3, "around $40": 3, "around $50": 3,
  "60+": 4, "$60+": 4, "fine dining": 4, "no limit": 4, "unlimited": 4, "no budget": 4,
};

function parsePriceCeiling(priceStr) {
  if (!priceStr) return null;
  const lower = priceStr.toLowerCase();
  for (const [key, level] of Object.entries(PRICE_BUDGET_MAP)) {
    if (lower.includes(key)) return level;
  }
  return null;
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

function getResultCap(results) {
  if (results.length === 0) return 0;
  const topScore = Math.max(...results.map(r => r.score));
  if (topScore >= 90) return 10;
  if (topScore >= 70) return 7;
  if (topScore >= 50) return 5;
  if (topScore >= 30) return 3;
  return 1;
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

async function fetchFromGoogle(query, analysis, userLat, userLng) {
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

  const allPlaces = new Map();

  for (const term of searchTerms) {
    // Don't append "restaurant" for brand/chain searches — it confuses Google Places
    const isBrandSearch = !analysis.cuisine && analysis.dish;
    // No longer hardcodes "Osaka Japan" — relies on the `location` bias param
    // (lat/lng) below, which now reflects the user's real position.
    const googleQuery = isBrandSearch
      ? `${term}${locationHint}`
      : `${term} restaurant${locationHint}`;
    console.log(`Google Places query: "${googleQuery}" (near ${lat}, ${lng})`);

    const params = {
      query: googleQuery,
      location: `${lat},${lng}`,
      radius: 5000,
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

    for (const place of (response.data.results || []).slice(0, 20)) {
      if (allPlaces.has(place.name)) continue;

      const placeLat = place.geometry?.location?.lat;
      const placeLng = place.geometry?.location?.lng;
      const distance = placeLat && placeLng ? getDistanceKm(lat, lng, placeLat, placeLng) : null;

      allPlaces.set(place.name, {
        name: place.name,
        cuisine: term,
        location: place.vicinity || "",
        price: priceLevel(place.price_level) || (place.rating ? `Rated ${place.rating}★` : "See Google Maps"),
        price_level: place.price_level ?? null,
        latitude: placeLat,
        longitude: placeLng,
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
      });
    }
  }

  return [...allPlaces.values()];
}

async function analyzeQuery(userQuery, previousQuery) {
  const context = previousQuery
    ? `The user previously searched: "${previousQuery}"\nNow they are refining with: "${userQuery}"`
    : `Query: "${userQuery}"`;

  const response = await client.messages.create({
    model: "claude-sonnet-5",
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
- If the user specifies a specific dish (e.g. "wagyu", "ramen", "carbonara"), set "dish" and set "cuisine" to null. Do NOT keep a generic cuisine tag when a specific dish has been named.
- If the user says "Japanese or Italian", set cuisine to "Japanese or Italian" exactly.
- Never infer a sub-cuisine (e.g. "sushi") from a broad term like "Japanese" — keep it broad.
- If the user names a specific restaurant chain or brand (e.g. "Starbucks", "McDonald's", "Ippudo"), set "dish" to that brand name so it gets searched directly on Google Places.
- If the user names a specific restaurant brand or chain (e.g. "Starbucks", "McDonald's", "Ippudo", "Ichiran"), set "dish" to that brand name and set "cuisine" to null. This ensures the brand name gets searched directly on Google Places without modification.

ATMOSPHERE RULES:
- "quiet", "relaxed", "peaceful", "chill", "unwind", "low-key" mean LOW NOISE and CASUAL PACE — NOT upscale, NOT omakase, NOT formal. Only use "upscale" or "fine dining" atmosphere if the user explicitly says those words.

Respond ONLY with a valid JSON object. No comments, no extra text, no markdown:
{
  "cuisine": "specific cuisine, 'Japanese or Italian', or null",
  "dish": "specific dish or null",
  "atmosphere": "vibe or null",
  "occasion": "occasion or null",
  "audience": "who they are dining with or null",
  "price": "budget as a plain phrase or null",
  "location": "specific sub-area or landmark within the city (e.g. Dotonbori, Shinsaibashi, Umeda) — NOT the city name itself. Null if no specific area mentioned.",  "priority": "the single most important thing in this query",
  "must_not": ["things explicitly NOT wanted — empty array if none"],
  "time_sensitive": true or false — true if query implies immediacy (tonight, now, hungry, for dinner, want to go). false for research/planning queries,
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

async function searchRestaurants(userQuery, analysis, restaurantData) {
  const priceCeiling = parsePriceCeiling(analysis.price);
  let filteredData = restaurantData;
  if (priceCeiling !== null && priceCeiling < 4) {
    const beforeCount = filteredData.length;
    filteredData = filteredData.filter(r =>
      r.price_level === null || r.price_level <= priceCeiling
    );
    console.log(`Price filter (≤ level ${priceCeiling}): ${beforeCount} → ${filteredData.length} restaurants`);
  }

  const response = await client.messages.create({
    model: "claude-sonnet-5",
    max_tokens: 4096, // raised from 2000 — Sonnet 5's tokenizer uses more tokens per
                       // response than older models, and 20 scored restaurant entries
                       // plus evidence/tags was very likely getting truncated at 2000.
    messages: [{
      role: "user",
      content: `You are a restaurant discovery AI.

The user searched: "${userQuery}"

Here is your analysis of what they want:
${JSON.stringify(analysis, null, 2)}

Here is the restaurant database:
${JSON.stringify(filteredData, null, 2)}

SCORING RULES:
- Score each restaurant from 0 to 100 based on how well it matches.
- CUISINE IS THE HIGHEST PRIORITY. Wrong cuisine = max score 35. Right cuisine = base score 60.
- If the user said "Japanese or Italian", both cuisines are equally valid — return a balanced mix, do NOT favour one over the other.
- If a specific dish was requested (e.g. wagyu, ramen), heavily weight restaurants likely to serve that dish.
- If a specific brand or chain was requested (e.g. Starbucks, McDonald's), the ONLY criteria is whether the restaurant name matches that brand. Score any matching location 90+. Ignore cuisine scoring entirely for brand searches.
- Factor in distance_km: under 1 km = great, under 3 km = good, under 5 km = acceptable.
- If a specific location or landmark was requested, prioritise restaurants closest to it.
- "must_not" items are HARD DISQUALIFIERS — score below 20 if matched.
- ATMOSPHERE: "quiet", "relaxed", "peaceful", "chill", "unwind", "low-key" mean LOW NOISE and CASUAL PACE. If atmosphere contains any of these words, HARD PENALISE omakase, counter dining, fine dining, and formal restaurants — score them below 40 regardless of rating. A neighbourhood izakaya or casual ramen shop should outscore a Michelin-starred counter.
- Use Google rating and review count as quality signals but do not let high ratings override a wrong vibe.
- Price has already been pre-filtered — do not penalise any restaurant in this list for price.
- Only return restaurants that genuinely match. Return [] if nothing scores above 30.

Respond ONLY with a JSON array, no other text:
[
  {
    "name": "Restaurant Name",
    "score": 92,
    "summary": "Why this matched (1-2 sentences, specific)",
    "evidence": ["evidence from description or reviews"],
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
  return JSON.parse(raw.slice(start, end + 1));
}

app.post("/search", async (req, res) => {
  const userQuery = req.body.query;
  const previousQuery = req.body.previousQuery || null;
  const userLat = req.body.latitude || null;
  const userLon = req.body.longitude || null;

  console.log("Search received:", userQuery);

  try {
    console.log("Calling analyzeQuery...");
    const analysis = await analyzeQuery(userQuery, previousQuery);
    console.log("Analysis done:", JSON.stringify(analysis));

    const isPivotOrNew = analysis.intent === "pivot" || analysis.intent === "new";
    const effectivePrevious = isPivotOrNew ? null : previousQuery;
    const fullQuery = effectivePrevious ? `${effectivePrevious}. Also: ${userQuery}` : userQuery;

    console.log("Fetching from Google Places...");
    const restaurantData = await fetchFromGoogle(fullQuery, analysis, userLat, userLon);
    console.log(`Got ${restaurantData.length} restaurants from Google`);

    console.log("Calling searchRestaurants...");
    const results = await searchRestaurants(fullQuery, analysis, restaurantData);

    results.forEach(r => {
      const match = restaurantData.find(d => d.name === r.name);
      if (match) {
        r.latitude = match.latitude;
        r.longitude = match.longitude;
        r.distance_km = match.distance_km;
        r.open_now = match.open_now;
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
    const isBrandQuery = (!analysis.cuisine || analysis.cuisine === '') && analysis.dish;
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
    const cap = getResultCap(sortedResults);
    const cappedResults = sortedResults.slice(0, cap);

    res.json({ analysis, results: cappedResults, intent: analysis.intent, clearedContext: isPivotOrNew });

  } catch (error) {
    console.error("Error:", error.message);
    res.status(500).json({ error: "Something went wrong" });
  }
});

app.listen(3000, () => {
  console.log("Server running at http://localhost:3000");
});