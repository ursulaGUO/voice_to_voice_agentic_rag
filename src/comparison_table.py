import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import os

# Cache for the CSV data and image lookup
_image_cache: Optional[pd.DataFrame] = None
_image_lookup: Optional[Dict[str, str]] = None
_product_url_lookup: Optional[Dict[str, str]] = None


def _load_csv_data() -> pd.DataFrame:
    """Load the CSV file and return as DataFrame."""
    global _image_cache
    
    if _image_cache is not None:
        return _image_cache
    
    try:
        # Find the CSV file
        project_root = Path(__file__).resolve().parents[1]
        csv_path = project_root / "marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"
        
        if not csv_path.exists():
            # Try alternative location
            csv_path = project_root / "data" / "marketing_sample_for_amazon_com-ecommerce__20200101_20200131__10k_data.csv"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Normalize column names
            df.columns = df.columns.str.lower().str.replace(" ", "_")
            _image_cache = df
            return df
    except Exception as e:
        print(f"Warning: Could not load CSV: {e}")
    
    return pd.DataFrame()


def _load_image_lookup() -> Dict[str, str]:
    """Load image URLs from CSV and create a lookup dictionary by uniq_id."""
    global _image_lookup
    
    if _image_lookup is not None:
        return _image_lookup
    
    _image_lookup = {}
    df = _load_csv_data()
    
    if not df.empty:
        # Create lookup: uniq_id -> first image URL
        for _, row in df.iterrows():
            uniq_id = str(row.get("uniq_id", "")).strip()
            image_str = str(row.get("image", "")).strip()
            
            if uniq_id and image_str and image_str != "nan":
                # Image URLs are pipe-separated, take the first one
                image_urls = image_str.split("|")
                if image_urls:
                    first_url = image_urls[0].strip()
                    if first_url and first_url.startswith("http"):
                        _image_lookup[uniq_id] = first_url
    
    return _image_lookup


def _load_product_url_lookup() -> Dict[str, str]:
    """Load product URLs from CSV and create a lookup dictionary by uniq_id."""
    global _product_url_lookup
    
    if _product_url_lookup is not None:
        return _product_url_lookup
    
    _product_url_lookup = {}
    df = _load_csv_data()
    
    if not df.empty:
        # Create lookup: uniq_id -> product URL
        for _, row in df.iterrows():
            uniq_id = str(row.get("uniq_id", "")).strip()
            product_url = str(row.get("product_url", "")).strip()
            
            if uniq_id and product_url and product_url != "nan" and product_url.startswith("http"):
                _product_url_lookup[uniq_id] = product_url
    
    return _product_url_lookup


def get_image_url(uniq_id: str) -> Optional[str]:
    """Get the first image URL for a product by its uniq_id."""
    if not uniq_id or uniq_id == "N/A":
        return None
    
    lookup = _load_image_lookup()
    return lookup.get(str(uniq_id).strip())


def get_product_url(uniq_id: str) -> Optional[str]:
    """Get the product URL for a product by its uniq_id."""
    if not uniq_id or uniq_id == "N/A":
        return None
    
    lookup = _load_product_url_lookup()
    return lookup.get(str(uniq_id).strip())


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
        uniq_id = result.get("uniq_id", "N/A")
        image_url = get_image_url(uniq_id) if uniq_id != "N/A" else None
        product_url = get_product_url(uniq_id) if uniq_id != "N/A" else None
        
        table_data.append({
            "Source": "Local Catalog",
            "Title": result.get("title", "N/A"),
            "Brand": result.get("brand", "N/A"),
            "Category": result.get("category", "N/A"),
            "Price": result.get("price", "N/A"),
            "Unique ID": uniq_id,
            "Image URL": image_url if image_url else "N/A",
            "URL": product_url if product_url else result.get("url", "N/A"),
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
            "Image URL": "N/A",
            "URL": result.get("url", "N/A"),
            "Score": "N/A"
        })
    
    df = pd.DataFrame(table_data)
    return df


def format_table_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as HTML table with images embedded."""
    if df.empty:
        return "No products found."
    
    # Create HTML table with images
    html = ['<table style="width:100%; border-collapse: collapse;">']
    
    # Header row
    headers = [col for col in df.columns if col != "Image URL"]
    if "Image URL" in df.columns:
        headers.insert(0, "Image")
    
    html.append('<thead><tr>')
    for header in headers:
        html.append(f'<th style="border: 1px solid #ddd; padding: 8px; text-align: left; background-color: #f2f2f2;">{header}</th>')
    html.append('</tr></thead>')
    html.append('<tbody>')
    
    # Data rows with images
    for idx, row in df.iterrows():
        html.append('<tr>')
        
        # Add image if available
        if "Image URL" in df.columns:
            image_url = row.get("Image URL", "N/A")
            if image_url and image_url != "N/A":
                html.append(f'<td style="border: 1px solid #ddd; padding: 8px;"><img src="{image_url}" style="max-width: 150px; height: auto;" /></td>')
            else:
                html.append('<td style="border: 1px solid #ddd; padding: 8px;">N/A</td>')
        
        # Add other columns (excluding Image URL)
        for col in df.columns:
            if col != "Image URL":
                value = str(row[col])
                # Make URLs clickable links
                if col == "URL" and value != "N/A" and value.startswith("http"):
                    value = f'<a href="{value}" target="_blank">{value[:50]}{"..." if len(value) > 50 else ""}</a>'
                else:
                    # Truncate long values
                    if len(value) > 50:
                        value = value[:47] + "..."
                html.append(f'<td style="border: 1px solid #ddd; padding: 8px;">{value}</td>')
        
        html.append('</tr>')
    
    html.append('</tbody></table>')
    return "\n".join(html)


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
