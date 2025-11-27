#!/usr/bin/env python3
"""
Windows FastAPI Server for Remote Query Execution System
Enhanced version with detailed result extraction AND raw output preservation
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, BackgroundTasks, status
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator
import uvicorn
import httpx

# ============================================================================
# Configuration
# ============================================================================

# Kali worker configuration
KALI_WORKER_URL = os.getenv("KALI_WORKER_URL", "http://192.168.56.103:9000")

# Storage configuration
DATA_DIR = Path("./data")
RESULTS_DIR = DATA_DIR / "results"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_OUTPUT_DIR = DATA_DIR / "raw_output"

# Create directories
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_LEVEL = logging.DEBUG if os.getenv("DEBUG", "false").lower() == "true" else logging.INFO
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(DATA_DIR / "server.log")
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# Data Models
# ============================================================================

class QueryRequest(BaseModel):
    """User query submission"""
    query: str = Field(..., min_length=1, max_length=500)
    amount: int = Field(default=10, ge=1, le=100)

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v):
        v = v.strip()
        if len(v) == 0:
            raise ValueError("Query cannot be empty")
        return v


class WebsiteMetadata(BaseModel):
    """Website metadata from darkdump"""
    viewport: Optional[str] = None
    generator: Optional[str] = None
    og_title: Optional[str] = None
    og_description: Optional[str] = None
    og_url: Optional[str] = None
    twitter_card: Optional[str] = None
    csrf_token: Optional[str] = None


class ProcessedWebsite(BaseModel):
    """Enhanced processed website entry"""
    title: str
    description: str
    onion_url: str
    clean_onion_domain: str
    keywords: List[str] = []
    sentiment: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None
    links_found: Optional[int] = None
    emails_found: Optional[int] = None
    documents_found: List[str] = []


class ProcessedResult(BaseModel):
    """Enhanced processed and cleaned result"""
    total_results: int
    websites: List[ProcessedWebsite]
    onion_domains: List[str]
    clearnet_domains: List[str]
    all_urls: List[str]
    all_emails: List[str]
    all_ips: List[str]
    document_links: List[str]
    keywords: List[str]
    statistics: Dict[str, Any]


class TaskResponse(BaseModel):
    """Response for query submission"""
    task_id: str
    status: str
    message: str
    timestamp: str
    query: str
    execution_time: float
    processed_results: Optional[ProcessedResult] = None
    raw_output_available: bool = True
    raw_json_available: bool = True


# ============================================================================
# Enhanced Result Processing Functions
# ============================================================================

class ResultProcessor:
    """Enhanced processor for darkdump results with detailed extraction"""
    
    @staticmethod
    def strip_ansi_codes(text: str) -> str:
        """Remove ANSI color codes from text"""
        ansi_pattern = r'\x1b\[[0-9;]*m'
        return re.sub(ansi_pattern, '', text)
    
    @staticmethod
    def clean_onion_domain(domain: str) -> str:
        """Clean and validate onion domain"""
        domain = domain.lower().strip().rstrip('.')
        
        # Remove trailing escape characters
        domain = re.sub(r'\x1b$', '', domain)
        
        # Ensure it ends with .onion
        if not domain.endswith('.onion'):
            if '.onion' in domain:
                domain = domain[:domain.index('.onion') + 6]
            else:
                domain += '.onion'
        
        return domain
    
    @staticmethod
    def parse_metadata(metadata_str: str) -> Dict[str, Any]:
        """Parse metadata JSON string"""
        try:
            # Clean the metadata string
            metadata_str = ResultProcessor.strip_ansi_codes(metadata_str)
            metadata = json.loads(metadata_str)
            return metadata
        except:
            return {}
    
    @staticmethod
    def parse_sentiment(sentiment_str: str) -> Optional[Dict[str, float]]:
        """Parse sentiment string"""
        try:
            # Extract Polarity and Subjectivity
            polarity_match = re.search(r'Polarity\s*=\s*([-\d.]+)', sentiment_str)
            subjectivity_match = re.search(r'Subjectivity\s*=\s*([-\d.]+)', sentiment_str)
            
            if polarity_match and subjectivity_match:
                return {
                    "polarity": float(polarity_match.group(1)),
                    "subjectivity": float(subjectivity_match.group(1))
                }
        except:
            pass
        return None
    
    @staticmethod
    def extract_websites_from_output(raw_output: str) -> List[Dict[str, Any]]:
        """Extract detailed structured website information from raw darkdump output"""
        websites = []
        
        # Clean ANSI codes first
        clean_output = ResultProcessor.strip_ansi_codes(raw_output)
        
        # Split by website entries using more flexible pattern
        # Match numbered entries with Website: line
        entry_pattern = r'(\d+\.[\s\S]*?(?=\n\d+\.|$))'
        entries = re.findall(entry_pattern, clean_output, re.MULTILINE)
        
        logger.info(f"Found {len(entries)} potential website entries")
        
        for entry in entries:
            try:
                website_data = {}
                
                # Extract title (after Website: label)
                title_match = re.search(r'Website:\s*(.+?)(?:\n|Information:|$)', entry)
                if title_match:
                    website_data['title'] = title_match.group(1).strip()
                else:
                    # Try alternative pattern
                    title_match = re.search(r'\[\+\]\s+Website:\s*(.+?)(?:\n|$)', entry)
                    if title_match:
                        website_data['title'] = title_match.group(1).strip()
                    else:
                        website_data['title'] = "Unknown"
                
                # Extract information/description
                info_match = re.search(r'Information:\s*(.+?)(?:\n.*?(?:Onion Link:|Keywords:|$))', entry, re.DOTALL)
                if info_match:
                    website_data['description'] = info_match.group(1).strip()
                else:
                    website_data['description'] = "No description provided"
                
                # Extract onion link
                onion_match = re.search(r'Onion Link:\s*(http://[a-z2-7]+\.onion[^\s\x1b]*)', entry)
                if onion_match:
                    website_data['onion_url'] = onion_match.group(1).strip()
                    # Extract clean domain
                    domain_match = re.search(r'http://([a-z2-7]+\.onion)', website_data['onion_url'])
                    if domain_match:
                        website_data['onion_domain'] = ResultProcessor.clean_onion_domain(domain_match.group(1))
                
                # Extract keywords
                keywords_match = re.search(r'Keywords?:\s*(.+?)(?:\n.*?(?:Sentiment:|Metadata:|Links Found:|$))', entry, re.DOTALL)
                if keywords_match:
                    keywords_str = keywords_match.group(1).strip()
                    # Split by comma and clean
                    keywords = [k.strip() for k in keywords_str.split(',') if k.strip()]
                    website_data['keywords'] = keywords
                else:
                    website_data['keywords'] = []
                
                # Extract sentiment
                sentiment_match = re.search(r'Sentiment:\s*(.+?)(?:\n|$)', entry)
                if sentiment_match:
                    website_data['sentiment'] = ResultProcessor.parse_sentiment(sentiment_match.group(1))
                
                # Extract metadata
                metadata_match = re.search(r'Metadata:\s*(\{.+?\})', entry, re.DOTALL)
                if metadata_match:
                    website_data['metadata'] = ResultProcessor.parse_metadata(metadata_match.group(1))
                
                # Extract links found count
                links_match = re.search(r'Links Found:\s*(\d+)', entry)
                if links_match:
                    website_data['links_found'] = int(links_match.group(1))
                
                # Extract emails found
                emails_match = re.search(r'Emails Found:\s*(.+?)(?:\n|$)', entry)
                if emails_match:
                    emails_str = emails_match.group(1).strip()
                    if emails_str.lower() != "no emails found.":
                        # Extract email addresses
                        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                        emails = re.findall(email_pattern, emails_str)
                        website_data['emails_found'] = emails
                
                # Extract documents found
                docs_match = re.search(r'Documents Found:\s*(.+?)(?:\n-{2,}|\n\d+\.|$)', entry, re.DOTALL)
                if docs_match:
                    docs_str = docs_match.group(1).strip()
                    if docs_str.lower() != "no document links found.":
                        # Extract URLs
                        url_pattern = r'http[s]?://[^\s,]+'
                        docs = re.findall(url_pattern, docs_str)
                        website_data['documents_found'] = docs
                
                # Only add if we have at least a title and onion URL
                if website_data.get('onion_url'):
                    websites.append(website_data)
                    logger.debug(f"Extracted website: {website_data.get('title', 'Unknown')}")
            
            except Exception as e:
                logger.warning(f"Error parsing website entry: {e}")
                continue
        
        logger.info(f"Successfully extracted {len(websites)} websites")
        return websites
    
    @staticmethod
    def extract_all_data(raw_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract all available data from raw result"""
        raw_output = raw_result.get('raw_output', '')
        darkdump_results = raw_result.get('darkdump_results', {})
        
        extracted = {
            'urls': set(),
            'onion_domains': set(),
            'clearnet_domains': set(),
            'emails': set(),
            'ips': set(),
            'keywords': set(),
            'document_links': set()
        }
        
        # From darkdump_results
        if darkdump_results:
            # URLs
            for url in darkdump_results.get('urls', []):
                clean_url = ResultProcessor.strip_ansi_codes(url).strip()
                clean_url = re.sub(r'\x1b$', '', clean_url)
                if clean_url and (clean_url.startswith('http://') or clean_url.startswith('https://')):
                    extracted['urls'].add(clean_url)
            
            # Domains
            for domain in darkdump_results.get('domains', []):
                clean_domain = ResultProcessor.clean_onion_domain(domain)
                if '.onion' in clean_domain and len(clean_domain) > 20:
                    extracted['onion_domains'].add(clean_domain)
                elif not domain.endswith('.') and '.' in domain:
                    extracted['clearnet_domains'].add(domain.lower().strip())
            
            # Emails
            for email in darkdump_results.get('emails', []):
                if '@' in email:
                    extracted['emails'].add(email.strip())
            
            # IPs
            for ip in darkdump_results.get('ips', []):
                extracted['ips'].add(ip.strip())
            
            # Keywords
            for keyword in darkdump_results.get('keywords', []):
                if keyword.strip():
                    extracted['keywords'].add(keyword.strip())
        
        # Additional extraction from raw output
        clean_output = ResultProcessor.strip_ansi_codes(raw_output)
        
        # Extract all onion URLs from output
        onion_pattern = r'http://[a-z2-7]{16,56}\.onion[^\s\x1b<>"{}|\\^`\[\]]*'
        for url in re.findall(onion_pattern, clean_output):
            clean_url = re.sub(r'\x1b$', '', url).strip()
            extracted['urls'].add(clean_url)
            # Extract domain
            domain_match = re.search(r'http://([a-z2-7]+\.onion)', clean_url)
            if domain_match:
                extracted['onion_domains'].add(domain_match.group(1))
        
        # Extract all emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        for email in re.findall(email_pattern, clean_output):
            extracted['emails'].add(email)
        
        # Extract document links
        doc_pattern = r'http[s]?://[^\s]+\.(?:pdf|doc|docx|txt|xml|json|csv)'
        for doc in re.findall(doc_pattern, clean_output, re.IGNORECASE):
            clean_doc = re.sub(r'\x1b$', '', doc).strip()
            extracted['document_links'].add(clean_doc)
        
        return {k: sorted(list(v)) for k, v in extracted.items()}
    
    @staticmethod
    def process_raw_result(raw_result: Dict[str, Any]) -> ProcessedResult:
        """Process raw result into comprehensive, structured format"""
        logger.info("Processing raw darkdump result with enhanced extraction")
        
        raw_output = raw_result.get('raw_output', '')
        
        # Extract structured website data from raw output
        websites_raw = ResultProcessor.extract_websites_from_output(raw_output)
        
        # Clean and structure websites
        processed_websites = []
        seen_urls = set()
        
        for site in websites_raw:
            if site['onion_url'] not in seen_urls:
                processed_websites.append(ProcessedWebsite(
                    title=site['title'],
                    description=site['description'],
                    onion_url=site['onion_url'],
                    clean_onion_domain=site.get('onion_domain', 'unknown.onion'),
                    keywords=site.get('keywords', []),
                    sentiment=site.get('sentiment'),
                    metadata=site.get('metadata'),
                    links_found=site.get('links_found'),
                    emails_found=site.get('emails_found', []) if isinstance(site.get('emails_found'), list) else None,
                    documents_found=site.get('documents_found', [])
                ))
                seen_urls.add(site['onion_url'])
        
        # Extract all additional data
        extracted_data = ResultProcessor.extract_all_data(raw_result)
        
        # Calculate comprehensive statistics
        statistics = {
            'total_websites': len(processed_websites),
            'onion_domains': len(extracted_data['onion_domains']),
            'clearnet_domains': len(extracted_data['clearnet_domains']),
            'total_urls': len(extracted_data['urls']),
            'onion_urls': len([u for u in extracted_data['urls'] if '.onion' in u]),
            'clearnet_urls': len([u for u in extracted_data['urls'] if '.onion' not in u]),
            'total_emails': len(extracted_data['emails']),
            'total_ips': len(extracted_data['ips']),
            'total_keywords': len(extracted_data['keywords']),
            'document_links': len(extracted_data['document_links']),
            'websites_with_metadata': len([w for w in processed_websites if w.metadata]),
            'websites_with_keywords': len([w for w in processed_websites if w.keywords]),
            'websites_with_sentiment': len([w for w in processed_websites if w.sentiment])
        }
        
        logger.info(f"Processing complete: {statistics['total_websites']} websites, "
                   f"{statistics['total_urls']} URLs, {statistics['total_emails']} emails")
        
        return ProcessedResult(
            total_results=len(processed_websites),
            websites=processed_websites,
            onion_domains=extracted_data['onion_domains'],
            clearnet_domains=extracted_data['clearnet_domains'],
            all_urls=extracted_data['urls'],
            all_emails=extracted_data['emails'],
            all_ips=extracted_data['ips'],
            document_links=extracted_data['document_links'],
            keywords=extracted_data['keywords'],
            statistics=statistics
        )


# ============================================================================
# In-Memory Task Storage
# ============================================================================

class TaskStore:
    """Simple in-memory task storage"""
    
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.lock = asyncio.Lock()
    
    async def create_task(self, task_id: str, query: str, amount: int) -> Dict[str, Any]:
        """Create a new task"""
        async with self.lock:
            timestamp = datetime.utcnow().isoformat() + "Z"
            
            self.tasks[task_id] = {
                "task_id": task_id,
                "query": query,
                "amount": amount,
                "status": "pending",
                "created_at": timestamp,
                "updated_at": timestamp,
                "raw_result": None,
                "processed_result": None,
                "raw_output": None,
                "execution_time": 0
            }
            
            logger.info(f"Created task {task_id} for query: {query}")
            return self.tasks[task_id]
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task by ID"""
        async with self.lock:
            return self.tasks.get(task_id)
    
    async def update_task(self, task_id: str, **updates):
        """Update task fields"""
        async with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id].update(updates)
                self.tasks[task_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    async def list_tasks(self) -> List[Dict[str, Any]]:
        """List all tasks"""
        async with self.lock:
            return list(self.tasks.values())


task_store = TaskStore()


# ============================================================================
# Kali Worker Communication
# ============================================================================

async def execute_on_kali(task_id: str, query: str, amount: int) -> Dict[str, Any]:
    """Execute query on Kali worker and get results"""
    logger.info(f"Sending query to Kali worker: {query}")
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{KALI_WORKER_URL}/execute",
                json={
                    "task_id": task_id,
                    "query": query,
                    "amount": amount
                }
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Received result from Kali worker for task {task_id}")
            return result
    
    except httpx.HTTPError as e:
        logger.error(f"HTTP error communicating with Kali worker: {e}")
        return {
            "success": False,
            "error": f"Failed to communicate with Kali worker: {str(e)}",
            "raw_output": "",
            "darkdump_results": {},
            "execution_time": 0
        }
    except Exception as e:
        logger.error(f"Error executing on Kali: {e}")
        return {
            "success": False,
            "error": str(e),
            "raw_output": "",
            "darkdump_results": {},
            "execution_time": 0
        }


async def store_results(task_id: str, raw_result: Dict[str, Any], processed_result: ProcessedResult, raw_output: str):
    """Store raw output, raw JSON result, and processed results"""
    try:
        timestamp = datetime.utcnow().isoformat().replace(':', '-').replace('.', '-')
        
        # Store raw text output (exactly as generated by darkdump)
        raw_output_filename = f"{timestamp}_{task_id}_raw_output.txt"
        raw_output_path = RAW_OUTPUT_DIR / raw_output_filename
        
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write(raw_output)
        
        logger.info(f"Stored raw text output at {raw_output_path}")
        
        # Store raw JSON result
        raw_filename = f"{timestamp}_{task_id}_raw.json"
        raw_path = RESULTS_DIR / raw_filename
        
        with open(raw_path, 'w', encoding='utf-8') as f:
            json.dump(raw_result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Stored raw JSON result at {raw_path}")
        
        # Store processed result
        processed_filename = f"{timestamp}_{task_id}_processed.json"
        processed_path = PROCESSED_DIR / processed_filename
        
        with open(processed_path, 'w', encoding='utf-8') as f:
            json.dump(processed_result.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Stored processed result at {processed_path}")
        
        return str(raw_output_path), str(raw_path), str(processed_path)
    
    except Exception as e:
        logger.error(f"Error storing results: {e}")
        return None, None, None


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Enhanced Remote Query Execution System",
    description="On-demand query distribution with comprehensive result processing + raw output",
    version="2.3.0"
)


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "version": "2.3.0",
        "kali_worker": KALI_WORKER_URL
    }


@app.post("/api/v1/query", response_model=TaskResponse)
async def submit_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """Submit a new query for immediate processing"""
    
    try:
        # Generate task ID
        task_id = f"task_{uuid4().hex[:16]}"
        
        # Create task
        await task_store.create_task(task_id, request.query, request.amount)
        
        # Execute on Kali worker immediately
        raw_result = await execute_on_kali(task_id, request.query, request.amount)
        
        # Process the result
        if raw_result.get("success"):
            # Extract raw output
            raw_output = raw_result.get('raw_output', '')
            
            # Process results
            processed_result = ResultProcessor.process_raw_result(raw_result)
            
            # Store results in background
            background_tasks.add_task(store_results, task_id, raw_result, processed_result, raw_output)
            
            # Update task
            await task_store.update_task(
                task_id,
                status="completed",
                raw_result=raw_result,
                processed_result=processed_result.model_dump(),
                raw_output=raw_output,
                execution_time=raw_result.get("execution_time", 0)
            )
            
            return TaskResponse(
                task_id=task_id,
                status="completed",
                message="Query executed and processed successfully",
                timestamp=datetime.utcnow().isoformat() + "Z",
                query=request.query,
                execution_time=raw_result.get("execution_time", 0),
                processed_results=processed_result,
                raw_output_available=True,
                raw_json_available=True
            )
        else:
            # Handle failure
            await task_store.update_task(
                task_id,
                status="failed",
                raw_result=raw_result,
                execution_time=raw_result.get("execution_time", 0)
            )
            
            return TaskResponse(
                task_id=task_id,
                status="failed",
                message=f"Query execution failed: {raw_result.get('error', 'Unknown error')}",
                timestamp=datetime.utcnow().isoformat() + "Z",
                query=request.query,
                execution_time=raw_result.get("execution_time", 0),
                processed_results=None,
                raw_output_available=False,
                raw_json_available=False
            )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@app.get("/api/v1/task/{task_id}")
async def get_task_result(task_id: str):
    """Get processed result for a specific task"""
    
    task = await task_store.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    return JSONResponse(content={
        "task_id": task["task_id"],
        "query": task["query"],
        "status": task["status"],
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "execution_time": task.get("execution_time", 0),
        "processed_results": task.get("processed_result"),
        "raw_output_available": task.get("raw_output") is not None,
        "raw_json_available": task.get("raw_result") is not None
    })


@app.get("/api/v1/task/{task_id}/raw")
async def get_task_raw_result(task_id: str):
    """Get raw JSON result for a specific task"""
    
    task = await task_store.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    raw_result = task.get("raw_result")
    if not raw_result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raw JSON result not available for task {task_id}"
        )
    
    return JSONResponse(content=raw_result)


@app.get("/api/v1/task/{task_id}/raw-output", response_class=PlainTextResponse)
async def get_task_raw_output(task_id: str):
    """Get exact raw text output from darkdump (as generated by the tool)"""
    
    task = await task_store.get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task {task_id} not found"
        )
    
    raw_output = task.get("raw_output")
    if not raw_output:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Raw output not available for task {task_id}"
        )
    
    return PlainTextResponse(content=raw_output)


@app.get("/api/v1/tasks")
async def list_all_tasks():
    """List all tasks with summary"""
    tasks = await task_store.list_tasks()
    
    # Create summary without full results
    task_summaries = []
    for task in tasks:
        processed = task.get("processed_result", {})
        statistics = processed.get("statistics", {}) if processed else {}
        
        summary = {
            "task_id": task["task_id"],
            "query": task["query"],
            "status": task["status"],
            "created_at": task["created_at"],
            "execution_time": task.get("execution_time", 0),
            "results_count": processed.get("total_results", 0) if processed else 0,
            "statistics": statistics,
            "raw_output_available": task.get("raw_output") is not None,
            "raw_json_available": task.get("raw_result") is not None
        }
        task_summaries.append(summary)
    
    return JSONResponse(content={
        "tasks": task_summaries,
        "total_tasks": len(task_summaries)
    })


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Remote Query Execution Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8443, help="Port to bind to")
    parser.add_argument("--kali-url", help="Kali worker URL")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.kali_url:
        KALI_WORKER_URL = args.kali_url
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"Starting Enhanced server on {args.host}:{args.port}")
    logger.info(f"Kali worker URL: {KALI_WORKER_URL}")
    logger.info(f"Results directory: {RESULTS_DIR.absolute()}")
    logger.info(f"Processed directory: {PROCESSED_DIR.absolute()}")
    logger.info(f"Raw output directory: {RAW_OUTPUT_DIR.absolute()}")
    logger.info("Authentication: DISABLED")
    logger.info("Mode: ON-DEMAND execution with comprehensive processing + raw output preservation")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port
    )