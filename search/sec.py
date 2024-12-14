import requests

def get_sec_links(symbol):
    base_url = f"https://api.nasdaq.com/api/company/{symbol.upper().replace('-','%25sl%25')}/sec-filings?sortColumn=filed&sortOrder=desc"
    print(f"Fetching data from: {base_url}")
    
    response = requests.get(base_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    body = response.json()
    
    if body["status"]["rCode"] != 200:
        raise Exception(f"Failed to fetch data for {symbol} - {body['status']['developerMessage']}")
    
    data = body["data"]["rows"]
    
    # Filter Latest Filings by Form Type
    filtered_filings = {}
    for row in data:
        form_type = row.get("formType")
        if form_type and form_type not in filtered_filings:
            filtered_filings[form_type] = row
    
    filterRowList=[]
    for data in filtered_filings.values():
        filterRowList.append(data)

        
    return filterRowList