import pandas as pd
from typing import List, Dict, Any


def create_comparison_table(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a product comparison table from search results."""
    if not results:
        return pd.DataFrame()
    
    # Separate local corpus and web search results
    local_results = [r for r in results if r.get("source") == "local_corpus"]
    web_results = [r for r in results if r.get("source") == "web_search"]
    
    table_data = []
    
    # Add local corpus results
    for result in local_results:
        table_data.append({
            "Source": "Local Catalog",
            "Title": result.get("title", "N/A"),
            "Brand": result.get("brand", "N/A"),
            "Category": result.get("category", "N/A"),
            "Price": result.get("price", "N/A"),
            "Unique ID": result.get("uniq_id", "N/A"),
            "URL": "N/A",
            "Score": f"{result.get('score', 0):.3f}" if result.get("score") else "N/A"
        })
    
    # Add web search results
    for result in web_results:
        table_data.append({
            "Source": "Web Search",
            "Title": result.get("title", "N/A"),
            "Brand": "N/A",
            "Category": "N/A",
            "Price": "N/A",
            "Unique ID": "N/A",
            "URL": result.get("url", "N/A"),
            "Score": "N/A"
        })
    
    df = pd.DataFrame(table_data)
    return df


def format_table_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as markdown table."""
    if df.empty:
        return "No products found."
    
    return df.to_markdown(index=False)


def format_table_text(df: pd.DataFrame) -> str:
    """Format DataFrame as readable text for TTS."""
    if df.empty:
        return "No products found."
    
    text_lines = []
    text_lines.append("Product Comparison Table:")
    text_lines.append("")
    
    for idx, row in df.iterrows():
        text_lines.append(f"Product {idx + 1}:")
        text_lines.append(f"  Source: {row['Source']}")
        text_lines.append(f"  Title: {row['Title']}")
        if row['Brand'] != "N/A":
            text_lines.append(f"  Brand: {row['Brand']}")
        if row['Category'] != "N/A":
            text_lines.append(f"  Category: {row['Category']}")
        if row['Price'] != "N/A":
            text_lines.append(f"  Price: ${row['Price']}")
        if row['Unique ID'] != "N/A":
            text_lines.append(f"  Unique ID: {row['Unique ID']}")
        if row['URL'] != "N/A":
            text_lines.append(f"  URL: {row['URL']}")
        text_lines.append("")
    
    return "\n".join(text_lines)
