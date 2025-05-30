from elasticsearch import Elasticsearch
import os
import pandas as pd
import time
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv
from time import strftime, gmtime
from datetime import datetime
from textwrap import dedent
from dateutil.relativedelta import relativedelta
from sentence_transformers import CrossEncoder

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Connect to Elasticsearch
es = Elasticsearch(
    "https://my-elasticsearch-project-c31d65.es.us-central1.gcp.elastic.cloud:443",
    api_key="bzRUUjFKWUJDOFpvVS1UeUlTY046T1VfVFJtTHBDc1M4M0RzZnFkcmhVdw=="
)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

index_name = "report_demo"

load_dotenv()

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

def retrieve_documents(query, top_k):
        start = time.time()  # Start time JUST for retrieval
        response = es.search(
            index=index_name,
            size=top_k,
            query={
                "match": {
                    "text": query
                }
            }
        )
        end = time.time()  # End time JUST for retrieval
        search_time = end - start

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "score": hit["_score"],
                "filename": hit["_source"]["filename"],
                "text": hit["_source"]["text"]
            })
        return results, search_time

# TOP_K=[10,50,100,200]
TOP_K=[10]

for top_k in TOP_K:

    questions = [
            "How does the IIP differ from the broader idea of national output, say during the period of covid pandemic?",
            "Why is the Inflation Expectations Survey of Households (IESH) conducted every two months, and what objectives does it serve?",
            "What were the reasons behind India experiencing relatively moderate inflation overshoot compared to other countries sometime last fiscal year?",
            "What was the impact on inflation dynamics during the period surrounding India’s demonetisation episode?"
    ]

    analytics = []

    for i, question in enumerate(questions):
        print(f"\n🔍 Question {i+1}: {question}")

        # Step 1: Rephrase the query using LLM
        llm_query = clarify_query(question)
        print(f"Rephrased Question {i+1}: {llm_query}")

        # Step 2: Retrieve top_k results using Elasticsearch
        raw_results, search_time = retrieve_documents(llm_query, top_k)
        print(f"Retrieved documents: {len(raw_results)}")
        print(f"ElasticSearch Retrieval Time: {search_time:.4f} seconds")

        if not raw_results:
            print("No documents retrieved from Elasticsearch.")
            continue

        try:
            query_date = fetch_date(llm_query).strip()
            query_min_date = fetch_min_date(llm_query).strip()
            query_duration = abs(months_since(query_min_date, query_date))
            window_size = 3 + min(12, query_duration)
        except:
            query_date = 'today'
            window_size = 6

        months_after = int(max(1, min(window_size, 2)))
        months_before = max(1, window_size - months_after)
        es_date_filter = build_range_around_date(query_date, months_before, months_after)["filter"]
        
        # Step 3: Prepare pairs for cross-encoder reranking
        top_n_results = raw_results[:top_k]  
        pairs = [(llm_query, item["text"]) for item in top_n_results]

        start_cross = time.time()
        scores = cross_encoder.predict(pairs)

        # For Elasticsearch, you don't have date in the results, so simulate deltas as 0 or ignore date boosting
        deltas = [0] * len(top_n_results)  

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
            date_boosts = [0 if mindelta <= delta <= maxdelta else 25 for delta in deltas]
            final_scores = [s - b for s, b in zip(scores, date_boosts)]

            reranked = sorted(
                zip(top_n_results, final_scores),
                key=lambda x: x[1],
                reverse=True
            )

            top_final = []
            for item, score in reranked[:top_k + 1]:
                if score > 2 - 2 * chunk_attempt:
                    item["cross_score"] = float(score)
                    top_final.append(item)
            end_cross = time.time()        
            if not top_final:
                print(f"No confident scores in attempt {chunk_attempt}, retrying with relaxed threshold.")
        

        # Step 5: Prepare final output
        content = ""
        for j, res in enumerate(top_final[:5]):
            content += f"Content{j+1} ({res["filename"]}): {res['text']}\n"

        analytics.append({
            "Question": question,
            "Content": content,
            "ES_Retrieval_Time": round(search_time, 4),
            "Cross_Rerank_Time": round(end_cross - start_cross, 4)
        })

        print(f"Cross_Rerank_Time: {round(end_cross - start_cross, 4)} seconds")
        print(f"Top-{top_k} Cross-Encoder reranked results recorded.")
        print("-------------------------------------------------------------------------------------------------------------")

    # Save results to CSV
    df = pd.DataFrame(analytics)
    # df.to_csv(f"analytics_with_crossenc_top{top_k}.csv", index=False)
    # print(f"\nanalytics_with_crossenc_top{top_k}.csv generated successfully.")
    df.to_csv(f"qualitative.csv", index=False)
    print(f"\nqualitative.csv generated successfully.")