import logging
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
from encoder import emb_text, model
from es_utils import get_search_results
import os
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from dateutil.relativedelta import relativedelta
from textwrap import dedent
from google import genai
from google.genai import types
from   google.genai.types import Tool, GenerateContentConfig, GoogleSearch
import time
from time import strftime, gmtime
import re
from typing import List, Dict
from elasticsearch import Elasticsearch

# Load environment variables
load_dotenv()

# API Key and Security
API_KEY = os.getenv("ACQ_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# FastAPI instance
app = FastAPI(title="Version 4 Server for Top 5 Search Results with URL and Summary")

# Milvus Configuration
# CPI_V4_COLLECTION_NAME = os.getenv("CPI_V4_COLLECTION_NAME")
# MILVUS_ENDPOINT = os.getenv("MILVUS_ENDPOINT")
# MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
# TOP_N_RESULTS = 5  # Configurable number of search results

# print(f'MILVUS_ENDPOINT = {MILVUS_ENDPOINT}')
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

# # Milvus client
# milvus_client = get_milvus_client(uri=MILVUS_ENDPOINT, token=MILVUS_TOKEN)

# Elasticsearch credentials
index_name=os.getenv("INDEX_NAME")
cloud_id=os.getenv("cloud_id")
es_key=os.getenv("es_key")
es = Elasticsearch(
    cloud_id,
    api_key=es_key
)

# Logging setup
logging.basicConfig(
    filename="result2.log",  # Log file name
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# API Key verification dependency
async def verify_api_key(api_key: str = Depends(api_key_header)):
    logging.info(f"Received API Key: {api_key[:4]}****")  # Mask API key for security
    if api_key != os.getenv("ACQ_API_KEY"):
        logging.warning(f"Unauthorized API access attempt with key: {api_key[:4]}****")
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Input model
class Question(BaseModel):
    question: str
    top_k: int


"""
def clarify_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(fYou are tasked with rephrasing the given query to make it easier for an SQL agent to pull the right data. ONLY IF NECESSARY, generate a rephrased natural language version of the query keeping in mind the following steps:
            1. Analyze the provided query for key entities. For example, if the query is about inflation in vegetables which is a type of food, then include the phrase "inflation in vegetables which is a type of food".
            2. Analyze the query for the existence of dates or time-related phrases.
                a. For example, if the query asks about GDP growth since December 2024, rephrase it as "GDP growth from [December 2024] to [{curdate}]".
                b. For example, if the query asks about IIP in the last six months, compute the date six months BEFORE [{curdate}] and add this information to the query.
                c. If the query asks about the last quarter, compute the last full quarter BEFORE [{curdate}] and add this information to the query.
                d. If the query asks about vague timelines such as "long term" without specifying dates, use the date five years ago from [{curdate}].
                e. If no date exists in the query use [{curdate}].
            3. IMPORTANT: Analyze the provided query for the existence of multiple entities. For example, if the query asks about "contribution of Maharashtra to total GDP", rewrite it as "GDP of Maharashtra and GDP of India".
            4. Wherever applicable, list exact sub-categories of products, or states within India. If no particular products, states, or other sub-categories are mentioned, include this information explicitly using phrases such as "for all of India" or "for food at a category level". If no specific sub-categories or states are required, mention "use * for pulling data only at category level" or "use * for pulling data at the country level" in the query.
            4. Always remember that all queries are related to India. If the word "India" is not mentioned in the query, include this in the rephrased query.
            5. IMPORTANT: If there is a comparison in the query between two values, mention both separately in the rephrased query. For example, "IIP of A compared to B" should be written as "We want the ratio of IIP of A, to the IIP of B" in the rephrased query.
            6. Do not include your reasoning trace in the rephrased query. Your output should contain only the information that is required from the Vector database.

            temperature=0.01,
            ),
        contents=query
    )

    return response.text
"""

def clarify_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    google_search_tool = Tool(
        google_search = GoogleSearch()
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are tasked with rephrasing the given query to make it easier for a vector agent to pull the right data. ONLY IF NECESSARY, generate a rephrased natural language version of the query keeping in mind the following steps:
            1. Analyze the provided query for key entities. For example, if the query is about inflation in vegetables which is a type of food, then include the phrase "inflation in vegetables which is a type of food".
            2. Extract key phrases from the query and rewrite them with emphasis. For example, if the query asks about "state-level growth", rewrite it to "STATE LEVEL GROWTH (GDP)".
            3. Analyze the query for the existence of dates or time-related phrases.
                a. For example, if the query asks about GDP growth since December 2024, rephrase it as "GDP growth from [December 2024] to [{curdate}]".
                b. For example, if the query asks about IIP in the last six months, compute the date six months BEFORE [{curdate}] and add this information to the query.
                c. If the query asks about the last quarter, compute the last full quarter BEFORE [{curdate}] and add this information to the query.
                d. If the query asks about vague timelines such as "long term" without specifying dates, use the date five years ago from [{curdate}].
                e. IMPORTANT: If no date exists in the query use a range from 2 months before [{curdate}] to [{curdate}].
                f. If the query contains an event without a date (for example, "COVID" or "51st meeting" or "the last world cup"), then use the google_search_tool to attach a date to the event.
                VERY IMPORTANT: DO NOT use the web search to add any extraneous information to the query, apart from the extracted date.
                g. If the date is after [{curdate}], use [{curdate}] instead.
            4. If there are any numeric counters such as 1st, 2nd, 3rd, rewrite it in words. For example, 51st should be rewritten as "fifty first".
            5. Always remember that all queries are related to India. If the word "India" is not mentioned in the query, include this in the rephrased query.
            6. Do not include your reasoning trace in the rephrased query. Your output should contain only the information that is required from the Vector database.
            """),
            tools=[google_search_tool],
            temperature=0.0,
            ),
        contents=query
    )

    return response.text

def fetch_date(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are a date extractor. Your job is to extract a clear time reference point in time from user queries, if one exists.
            Examples:
            - "CPI report for December 2024" → "December 2024"
            - "What was the inflation rate in June 2023?" → "June 2023"
            - "Give me the latest IIP data" → "today"
            - "Tell me the GDP growth over the last five years" → "today"
            - "What happened in Q3 2022?" → "December 2022"
            - "What was the inflation rate H1 FY25?" → "September 2024"
            Return only the extracted date phrase (like "December 2024" or "today"). If no specific date is found, assume "today".
            IMPORTANT: If the extracted date contains multiple dates, output ONLY the LATEST date mentioned in the query.
            Make sure the output format of the date is "%B %Y"
            IMPORTANT: Do not output a date beyond {curdate}
            """),
            temperature=0.0,
        ),
        contents=query
    )

    query_date = response.text.strip()
    return query_date

def fetch_min_date(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    curdate = strftime("%Y-%m", gmtime())
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""You are a date extractor. Your job is to extract a clear time reference point in time from user queries, if one exists.
            Examples:
            - "CPI report for December 2024" → "December 2024"
            - "What was the inflation rate in June 2023?" → "June 2023"
            - "Give me the latest IIP data" → "today"
            - "Tell me the GDP growth over the last five years" → "today"
            - "What happened in Q3 2022?" → "December 2022"
            - "What was the inflation rate H1 FY25?" → "September 2024"
            Return only the extracted date phrase (like "December 2024" or "today"). If no specific date is found, assume "today".
            IMPORTANT: If the extracted date contains multiple dates, output ONLY the EARLIEST date mentioned in the query.
            Make sure the output format of the date is "%B %Y"
            IMPORTANT: Do not output a date beyond {curdate}
            """),
            temperature=0.0,
        ),
        contents=query
    )

    query_date = response.text.strip()
    return query_date


def months_since(date_str, query_date='today'):
    #Date from the query if not use latest
    try:
        date_obj = datetime.strptime(date_str, '%B %Y')
        if query_date == 'today':
            today = datetime.today()
        else:
            try:
                today = datetime.strptime(query_date, '%B %Y')
            except:
                today = datetime.today()  # fallback

        diff = relativedelta(today, date_obj)
        logging.info("Pulled date: " + query_date + ", Doc date: " + date_str + ", Delta: " + str(diff.years * 12 + diff.months))
        return diff.years * 12 + diff.months

    except Exception as e:
        print(f'Error parsing date: {e}')
        return 999

def build_range_around_date(center_date_str, months_before, months_after, field_name="date"):
    """
    Given a center date string like 'March 2024' and two integers (months_before, months_after),
    builds a Milvus filter expression for that range.

    Assumes Milvus stores dates in the format 'Month YYYY' (e.g., 'March 2024').
    """
    if center_date_str == 'today':
        center_date_str = datetime.today().strftime("%B %Y")
    try:
        center_date = datetime.strptime(center_date_str, "%B %Y")
    except:
        center_date_str = datetime.today().strftime("%B %Y")
        center_date = datetime.strptime(center_date_str, "%B %Y")

    # Calculate start and end of the range
    start_date = center_date - relativedelta(months=months_before)
    end_date = center_date + relativedelta(months=months_after)

    # Generate the list of months
    current = start_date
    months = []
    while current <= end_date:
        months.append(current.strftime("%B %Y"))
        current += relativedelta(months=1)

    # Build the filter expression
    filters = [f'{field_name} == "{m}"' for m in months]
    filter_expr = " or ".join(filters)

    return {"filter": filter_expr}

def generalize_query(query):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent("""Consider the query given in the content. Your task is to generalize the query to a small extent. You can do this by:
            1. Removing mentions of specific states of India and replacing it by just "India". For example, "Tamil Nadu" can be replaced by "India".
            2. Removing specific day-month references and retaining only the years. For example, "23 May 2023" can be replaced by just "2023".
            3. Removing names of specific commodities and replacing them by broader classes. For example, "moong dal" can be replaced by "pulses" or "food".
            4. Removing generic phrases and focusing only on the data. For example, "Effect of COVID on steel production" can be replaced by "steel production statistics". Similarly, phrases such as "commentary", "effect", "policy" can be removed from the rephrased query.
            INSTRUCTIONS: Return the rephrased query as a single sentence. Do not hallucinate non-existing information. Stick to a maximum length of 12 words.
            """),
            temperature=0.0,
            ),
        contents=query
    )
    logging.info("Rephrased query: " + response.text)
    return response.text

def suggest_answer(query, excerpts):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Consider the following query: {query}. You are given the following content that contains a potential answer for this query. Write a short paragraph or set of bullet points that summarize the answer to the query.

            Here is some brief context about aspects of the Indian economy:
                1. Agriculture: Contributes about 15-20% to GDP, agriculture employs nearly half of the workforce. Monsoon patterns significantly impact agricultural output and rural demand.
                2. Industry: Manufacturing and construction are crucial for GDP growth and employment. Government initiatives like "Make in India" aim to boost manufacturing.
                3. Services: The services sector, including IT, finance, and telecommunications, contributes over 50% to GDP. IT services, in particular, are a major export and growth driver.
                4. Government Policies: Fiscal policies, such as taxation and public spending, influence economic growth. Monetary policies by the Reserve Bank of India (RBI) manage inflation and interest rates.
                5. Inflation: Influenced by food prices, fuel costs, and global commodity prices. The RBI uses repo rates and cash reserve ratios to control inflation.
                6. Foreign Direct Investment (FDI): FDI inflows boost infrastructure, technology, and job creation. Government policies aim to attract FDI in sectors like manufacturing and services.
                7. Global Economic Conditions: Exports, remittances, and foreign investments are affected by global demand and economic stability.
                8. Demographics: A young and growing workforce can drive economic growth, but requires adequate education and employment opportunities.
                9. Infrastructure: Investments in transportation, energy, and digital infrastructure enhance productivity and economic growth.
                10. Technological Advancements: Innovation and digitalization improve efficiency and competitiveness across sectors.

            IMPORTANT: Include ONLY the information that answers the query. If there is insufficient information to answer the specific query, say exactly the following in your response: "<insufficient-data>". Do not include any other text when there is insufficient information to answer the query.
            **Formatting instructions**
            - Make your answer about 300 to 350 words, IF sufficient data is present.
            - Use each of the excerpts to compose your answer, so long as they are relevant to the query.
            - Do not mention any data that is not available or not present. Stick to a summary of what data is available.

            IMPORTANT: In your summary, focus on specific data values, prices, and percentages if they relate to the original query. Avoid returning purely qualitative observations.
            IMPORTANT: If you are returning a full answer, do not include "<insufficient-data>" in the response.
            IMPORTANT: DO NOT HALLUCINATE ANY INFORMATION THAT IS NOT PRESENT IN THE ATTACHED CONTENTS.
            """),
            temperature=0.0,
            ),
        contents=excerpts
    )
    return response.text

def get_reference_url(ref: str) -> str:
    if re.match(r"Inflation Expectations Survey of Households \w+ \d{4}", ref):
        return "https://website.rbi.org.in/web/rbi/statistics/survey?category=24927098&categoryName=Inflation%20Expectations%20Survey%20of%20House-holds%20-%20Bi-monthly"
    elif re.match(r"Monetary Policy Report \w+ \d{4}", ref):
        return "https://website.rbi.org.in/web/rbi/publications/articles?category=24927873"
    elif re.match(r"Minutes of the Monetary Policy Committee Meeting \w+ \d{4}", ref):
        return "https://website.rbi.org.in/web/rbi/press-releases?q=%22Minutes+of+the+Monetary+Policy+Committee+Meeting%22"
    elif re.match(r"CPI Press Release \w+ \d{4}", ref):
        return "https://www.mospi.gov.in/archive/press-release?field_press_release_category_tid=120"
    elif re.match(r"Economic Survey \d{4} ?- ?\d{4}", ref):
        return "https://www.indiabudget.gov.in/economicsurvey/allpes.php"
    elif re.match(r"IIP Press Release \w+ \d{4}", ref):
        return "https://www.mospi.gov.in/archive/press-release?field_press_release_category_tid=121"
    elif re.match(r"Monthly Economic Report \w+ \d{4}", ref):
         return "https://dea.gov.in/monthly-economic-report-table"
    elif re.match(r"RBI Bulletin \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/BS_ViewBulletin.aspx"
    elif re.match(r"RBI State Finances \w+ \d{4}", ref):
         return "https://rbi.org.in/Scripts/AnnualPublications.aspx?head=State%20Finances%20:%20A%20Study%20of%20Budgets"
    else:
        return "Unknown Url"  # Default empty if no match

def synthesize_with_gemini(
    question: str,
    unstructured_results: List[Dict]
) -> str:
    """
    Synthesizes a final answer using Gemini-2.0-Flash based on unstructured sources.
    """

    # Format top results
    formatted_sources = ""
    for idx, result in enumerate(unstructured_results, start=1):
        content = result.get("content", "").strip()
        reference = result.get("reference", "").strip()
        url = result.get("url", "").strip()

        formatted_sources += (
            f"### Source {idx}:\n"
            f"- **Reference**: {reference}\n"
            f"- **URL**: {url if url else 'N/A'}\n"
            f"- **Extracted Content**:\n{content}\n\n"
        )

    # System instruction prompt without structured data logic
    client = genai.Client(api_key=GOOGLE_API_KEY)
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        config=types.GenerateContentConfig(
            system_instruction=dedent(f"""Based on the original question: {question}, and the following unstructured text data from various sources, synthesize a comprehensive and coherent answer. Integrate the information smoothly.

        **Formatting Instructions:**
        - Begin the final answer with this header: `## Insights from Ingested Data`
        - **When using any content from the 'Unstructured Text Data', cite the corresponding 'Reference' name and URL, but only if it is actually used.**
        - Indicate citations inline using square brackets like this: [1], [2], etc.
        - At the end of the answer, add a section titled `## References` that lists only the used references, numbered to match the inline citations.
        - Only include references that were cited in the text.
        - If no URL is available, skip the reference entirely.
        - Format the `## References` section exactly like this:

        ## References
        1. [Reference Name 1](https://example.com)
        2. [Reference Name 2](https://example.com)

        - Do not include any references that were not cited in the synthesized answer.
        - Avoid duplicate citations for the same source in the same paragraph — cite once per distinct point.
        - If data is conflicting or ambiguous, acknowledge that transparently in the summary.

            """),
            temperature=0.0,
            ),
        contents=formatted_sources
    )
    return response.text


@app.post("/search-topN", dependencies=[Depends(verify_api_key)])
async def search_topN_es(request: Request, data: Question):
    start_time = time.time()
    request_time = datetime.utcnow().isoformat()

    llm_query = clarify_query(data.question).strip()

    try:
        query_date = fetch_date(llm_query).strip()
        query_min_date = fetch_min_date(llm_query).strip()
        query_duration = abs(months_since(query_min_date, query_date))
        window_size = 3 + min(12, query_duration)
        logging.info(f"Query min date: {query_min_date}, max date: {query_date}, Query duration is {query_duration}")
    except:
        query_date = 'today'
        window_size = 6

    months_after = int(max(1, min(window_size, 2)))
    months_before = max(1, window_size - months_after)
    es_date_filter = build_range_around_date(query_date, months_before, months_after)["filter"]

    client_ip = request.client.host  # Get client IP address

    logging.info(f"Received request from {client_ip} at {request_time}")
    logging.info(f"Question Asked: {data.question}")
    logging.info(f"LLM Query Generated: {llm_query}")

    try:
        token_count = len(data.question.split())  # Approximate token count
        logging.info(f"Question Token Count: {token_count}")

        llm_token_count = len(llm_query.split())  # Approximate token count
        logging.info(f"LLM Query Token Count: {llm_token_count}")

        embed_start = time.time()
        query_vector = emb_text(model, llm_query)  # encodes to 768-dim list using all-mpnet-base-v2
        embed_time = time.time() - embed_start
        logging.info(f"Embedding generation time: {embed_time:.4f} seconds")

        search_start = time.time()
        logging.info("Elasticsearch Index: " + index_name)

        search_res = get_search_results(
            es, index_name, query_vector, ["text", "filename"], data.top_k
        )

        print("Number of chunks before Re-ranking:",len(search_res[0]))
        
        search_time = time.time() - search_start
        logging.info(f"Elasticsearch search execution time: {search_time:.4f} seconds")
        logging.info(f"Document search date filter: {es_date_filter}")

        if not search_res or not search_res[0]:
            logging.warning("No results found for query")
            raise HTTPException(status_code=404, detail="No results found")

        top_N = data.top_k
        
        top_n_results = [
            {
                "content": result["entity"]["text"],
                "distance": result["distance"],
                "source": result["entity"].get("filename", ""),
                "page": None,
                "reference": None,
                "date": None,
                "url": None
            }
            for result in search_res[0][:top_N]
        ]

        print("Number of chunks after Re-ranking:",len(top_n_results))
        
        logging.info(f"Top {top_N} results returned directly from Elasticsearch:")
        for i, res in enumerate(top_n_results, start=1):
            logging.info(
                f"{i}. Content: {res['content'][:200]}..., Page: {res['page']}, "
                f"Source: {res['source']}, Reference: {res['reference']}, Date: {res['date']}, Distance: {res['distance']:.4f}"
            )

        # --- Begin Cross-encoder reranking logic ---

        start_cross = time.time()

        # Build pairs for cross-encoder: (query, document content)
        pairs = [(llm_query, item["content"]) for item in top_n_results]
        scores = cross_encoder.predict(pairs)

        # For Elasticsearch, you don't have date in the results, so simulate deltas as 0 or ignore date boosting
        deltas = [0] * len(top_n_results)  # or you can implement if you have date info

        # Use window_size to compute date boost range, but here deltas all 0 so date boosting is neutral
        if min(deltas) > 0:
            maxdelta = min(deltas)
        elif max(deltas) < 0:
            maxdelta = max(deltas)
        else:
            maxdelta = 0
        maxdelta += 0.5 * window_size
        mindelta = maxdelta - window_size

        top_final = []
        chunk_attempt = 0
        while not top_final and chunk_attempt < 2:
            chunk_attempt += 1
            logging.info(f"Using deltas range: [{mindelta}, {maxdelta}]")
            date_boosts = [0 if mindelta <= delta <= maxdelta else 25 for delta in deltas]
            final_scores = [s - b for s, b in zip(scores, date_boosts)]

            reranked = sorted(
                zip(top_n_results, final_scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_final = []
            for item, score in reranked[:top_N + 1]:
                if score > 2 - 2 * chunk_attempt:
                    item["cross_score"] = float(score)
                    top_final.append(item)

            if not top_final:
                logging.warning(f"Not enough high-confidence results found ({len(top_final)}/{top_N}). Relaxing filters...")
                mindelta += 2
                maxdelta += 2

        end_cross = time.time()

        if not top_final:
            logging.warning("No valid results with cross_score > threshold.")
            return {
                "question": data.question,
                "llm_query": llm_query,
                "query_date": query_date,
                "retrieved_results": [{
                    "content": "We could not find any relevant content related to your query.",
                    "distance": "N/A",
                    "source": "N/A",
                    "page": "N/A",
                    "reference": "N/A",
                    "date": "N/A",
                    "url": "N/A"
                }],
                "time": time.time() - start_time,
            }

        logging.info(f"Top-{top_N} results after reranking:")
        for i, res in enumerate(top_final[:top_N], 1):
            logging.info(
                f"{i}. Content: {res['content'][:200]}..., Page: {res['page']}, "
                f"Source: {res['source']}, Reference: {res['reference']}, Date: {res['date']}, "
                f"Distance: {res['distance']:.4f}, Cross Score: {res['cross_score']:.4f}"
            )

        total_time = time.time() - start_time

        return {
            "question": data.question,
            "llm_query": llm_query,
            "query_date": query_date,
            "retrieved_results": top_final,
            "embed_time": embed_time,
            "es_search_time": search_time,
            "rerank_time": end_cross - start_cross,
            "total_time": total_time,
        }

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        logging.error(error_message, exc_info=True)
        raise HTTPException(status_code=500, detail=error_message)
