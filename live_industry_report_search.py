from live_search.industry_reports import get_industry_reports
from live_search.pdf import get_pdf_content_nodes
from concurrent.futures import ThreadPoolExecutor


def handle_live_industry_reports_search(query):
    try:
        industry_report_files = get_industry_reports(query)

        nodes = []
        
        with ThreadPoolExecutor() as executor:
            file_nodes_list = list(executor.map(get_pdf_content_nodes, industry_report_files))
            for file_nodes in file_nodes_list:
                nodes.extend(file_nodes)
            
        return nodes
        
    except Exception as e:
        print(e)

        return []