from elasticsearch import Elasticsearch
import os
import pandas as pd
import time
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from dotenv import load_dotenv
from time import strftime, gmtime
from textwrap import dedent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Connect to Elasticsearch
es = Elasticsearch(
    "https://my-elasticsearch-project-c31d65.es.us-central1.gcp.elastic.cloud:443",
    api_key="bzRUUjFKWUJDOFpvVS1UeUlTY046T1VfVFJtTHBDc1M4M0RzZnFkcmhVdw=="
)

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

    # questions = [
    #         "Why is the Inflation Expectations Survey of Households (IESH) conducted every two months, and what objectives does it serve?",
    #         "What were the reasons behind India experiencing relatively moderate inflation overshoot compared to other countries sometime last fiscal year?",
    #         "Roughly how many urban households took part in the IESH survey conducted around 6 months ago?",
    #         "What impact did monetary tightening and policy steps taken sometime last year have on inflation and domestic demand?",
    #         "How does the IIP differ from the broader idea of national output, say during the period of recent economic recovery?"
    # ]
    questions = [
            "How does the IIP differ from the broader idea of national output, say during the period of covid pandemic?",
            "Why is the Inflation Expectations Survey of Households (IESH) conducted every two months, and what objectives does it serve?",
            "What were the reasons behind India experiencing relatively moderate inflation overshoot compared to other countries sometime last fiscal year?",
            "What was the impact on inflation dynamics during the period surrounding India’s demonetisation episode?"
    ]

    analytics = []

    # Iterate through questions
    for i, question in enumerate(questions):
        llm_query=clarify_query(question)
        results, search_time = retrieve_documents(question,top_k)

        print(f"\nQuestion {i+1}: {question}")
        print(f"Rephrased Question {i+1}: {llm_query} ")
        print("Retrieved documents:", len(results))
        print(f"ElasticSearch Retrieval Time: {search_time:.4f} seconds")

        if results:
            content = ""
            for j, res in enumerate(results[:5]):
                content += f"Content{j+1} ({res["filename"]}): {res['text']}\n"

            analytics.append({
                "Question": question,
                "Content": content,
                "ES_Retrieval_Time": round(search_time, 4)  # Save for comparison
            })
            print(f"Response recorded.")
        else:
            print(f"No results found.")
            
        print("-------------------------------------------------------------------------------------------------------------")    

    # Save results to CSV
    df = pd.DataFrame(analytics)
    # df.to_csv(f"analytics_without_cenc_top{top_k}.csv", index=False)
    # print(f"\n analytics_without_cenc_top{top_k}.csv generated successfully")
    df.to_csv(f"qualitative_es_woeb_woen.csv", index=False)
    print(f"\nqualitative.csv generated successfully.")
