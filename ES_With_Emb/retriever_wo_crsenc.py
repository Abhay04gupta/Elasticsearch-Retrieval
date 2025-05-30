import requests
import time
import pandas as pd

url = "http://127.0.0.1:8000/search-topN"
headers = {
    "Content-Type": "application/json",
    "access_token": "api-12345"
}

Num_chunks=[10]

for size in Num_chunks:

    # questions = questions = [
    #     "Why is the Inflation Expectations Survey of Households (IESH) conducted every two months, and what objectives does it serve?",
    #     "What were the reasons behind India experiencing relatively moderate inflation overshoot compared to other countries sometime last fiscal year?",
    #     "Roughly how many urban households took part in the IESH survey conducted around 6 months ago?",
    #     "What impact did monetary tightening and policy steps taken sometime last year have on inflation and domestic demand?",
    #     "How does the IIP differ from the broader idea of national output, say during the period of recent economic recovery?"
    # ]
    questions = [
            "How does the IIP differ from the broader idea of national output, say during the period of covid pandemic?",
            "Why is the Inflation Expectations Survey of Households (IESH) conducted every two months, and what objectives does it serve?",
            "What were the reasons behind India experiencing relatively moderate inflation overshoot compared to other countries sometime last fiscal year?",
            "What was the impact on inflation dynamics during the period surrounding India’s demonetisation episode?"
    ]

    analytics = []

    for idx, question in enumerate(questions, 1):
        data = {
            "question": question,
            "top_k": size
        }

        print(f"\nSending Question {idx}: {question}")
        start_time = time.time()
        response = requests.post(url, headers=headers, json=data)
        end_time = time.time()
        total_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            backend_time = result.get("time_taken", "N/A")
            es_seacrh_time=result.get("es_search_time","N/A")
            embed_time=result.get("embed_time","N/A")
            retrieved_results = result.get("retrieved_results", [])
            combined_content = ""

            for i, item in enumerate(retrieved_results[:3], 1):
                content = item.get("content", "N/A")
                source= item.get("source","N/A")
                combined_content += f"Content {i} (Source: {source}):\n{content}\n"

            analytics.append({
                "question": question,
                "combined_contents": combined_content.strip(),
                "embedding generation time": embed_time,
                "es_seacrh_time":es_seacrh_time,
                "backend_time (s)": backend_time,
                "total_time (s)": total_time
            })

            print(f"✅ Response received in {total_time:.2f} seconds")
            print(f"Backend time (without network latency): {backend_time} seconds")
        else:
            print(f"❌ Failed with status code: {response.status_code}")
            try:
                print("Details:", response.json())
            except Exception:
                print("No JSON response returned.")

        print("-----------------------------------------------------------------------------------------------------------------------------")

    # Exporting to CSV (optional)
    df = pd.DataFrame(analytics)
    # df.to_csv(f"analytics_top{size}_without_cen.csv", index=False)
    # print(f"📁 analytics_top{size}_without_cen.csv")
    df.to_csv(f"qualitative_es_woen.csv", index=False)
    print(f"qualitative_es_woen.csv.csv")