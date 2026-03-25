API_KEY = "AIzaSyB5zTW5tksNaqb90pqiDezgTz8wtiABl1s"

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    raise SystemExit(1)

print("Testing Gemini API connection...")

try:
    client = genai.Client(api_key=API_KEY)

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents="Say hello in one sentence.",
        config=types.GenerateContentConfig(
            max_output_tokens=64,
            temperature=0.1,
        ),
    )

    print("SUCCESS:", resp.text.strip())

except Exception as e:
    print("FAILED:", e)
