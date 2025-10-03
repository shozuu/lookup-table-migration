import os
import json
import time
import uuid
from typing import List, Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
from supabase import create_client, Client
import google.generativeai as genai
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('migration.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not all([SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY]):
    raise ValueError("Missing required environment variables. Check your .env file.")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)

def get_embedding(text: str, retries=5, backoff=2) -> List[float]:
    """Generate embedding for given text with retry logic."""
    for attempt in range(retries):
        try:
            logger.debug(f"Generating embedding for text (length: {len(text)})")
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.warning(f"Embedding generation error: {e} (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                raise e

def make_embedding_text(entry: Dict[str, Any]) -> str:
    """Create clean, pure content text for better embeddings."""
    sentences = []
    
    # Start with diagnosis and definition
    diagnosis = entry.get("diagnosis", "").strip().lower()
    definition = entry.get("definition", "").strip().lower()
    
    if diagnosis and definition:
        sentences.append(f"{diagnosis}. {definition}")
    elif diagnosis:
        sentences.append(diagnosis)
    
    # Add characteristics (pure content)
    if entry.get("defining_characteristics") and isinstance(entry["defining_characteristics"], list):
        characteristics = [char.strip().lower() for char in entry["defining_characteristics"] 
                         if char and char.strip() and char.strip().lower() != "to be developed"]
        if characteristics:
            sentences.append(", ".join(characteristics))
    
    # Combine all factors (pure content)
    all_factors = []
    
    if entry.get("related_factors") and isinstance(entry["related_factors"], list):
        related = [factor.strip().lower() for factor in entry["related_factors"] 
                  if factor and factor.strip() and factor.strip().lower() != "to be developed"]
        all_factors.extend(related)
    
    if entry.get("risk_factors") and isinstance(entry["risk_factors"], list):
        risks = [factor.strip().lower() for factor in entry["risk_factors"] 
                if factor and factor.strip() and factor.strip().lower() != "to be developed"]
        all_factors.extend(risks)
    
    if all_factors:
        sentences.append(", ".join(all_factors))
    
    # Add conditions (pure content)
    if entry.get("associated_conditions") and isinstance(entry["associated_conditions"], list):
        conditions = [cond.strip().lower() for cond in entry["associated_conditions"] 
                     if cond and cond.strip() and cond.strip().lower() != "to be developed"]
        if conditions:
            sentences.append(", ".join(conditions))
    
    # Add populations (pure content)
    if entry.get("at_risk_population") and isinstance(entry["at_risk_population"], list):
        populations = [pop.strip().lower() for pop in entry["at_risk_population"] 
                      if pop and pop.strip() and pop.strip().lower() != "to be developed"]
        if populations:
            sentences.append(", ".join(populations))
    
    return ". ".join(sentences) + "."

def check_existing_entry(diagnosis: str) -> bool:
    """Check if entry already exists in database to avoid duplicates."""
    try:
        result = supabase.table("nnn_lookup").select("id").eq("diagnosis", diagnosis).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.warning(f"Error checking existing entry: {e}")
        return False

def insert_row(entry: Dict[str, Any], embedding: List[float], retries=5, backoff=2) -> bool:
    """Insert row into Supabase with retry logic."""
    for attempt in range(retries):
        try:
            data = {
                "id": str(uuid.uuid4()),
                "diagnosis": entry.get("diagnosis", ""),
                "definition": entry.get("definition", ""),
                "defining_characteristics": entry.get("defining_characteristics", []),
                "related_factors": entry.get("related_factors", []),
                "risk_factors": entry.get("risk_factors", []),
                "at_risk_population": entry.get("at_risk_population", []),
                "associated_conditions": entry.get("associated_conditions", []),
                "suggested_outcomes": entry.get("suggested_outcomes", []),
                "suggested_interventions": entry.get("suggested_interventions", []),
                "embedding": embedding,
            }
            
            supabase.table("nnn_lookup").insert(data).execute()
            logger.debug(f"Successfully inserted: {entry.get('diagnosis', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.warning(f"DB insert error: {e} (attempt {attempt+1}/{retries})")
            if attempt < retries - 1:
                time.sleep(backoff ** attempt)
            else:
                logger.error(f"Failed to insert after {retries} attempts: {entry.get('diagnosis', 'Unknown')}")
                return False
    
    return False

def save_backup_with_embeddings(entries_with_embeddings: List[Dict[str, Any]], filename: str = "new_nnn_content_with_embeddings.json"):
    """Save backup file with embeddings for tracking purposes."""
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(entries_with_embeddings, f, indent=2, ensure_ascii=False)
        logger.info(f"Backup file saved: {filename}")
    except Exception as e:
        logger.error(f"Failed to save backup file: {e}")

def create_table_if_not_exists():
    """Ensure the nnn_lookup table exists with proper schema."""
    try:
        # Test if table exists by trying to select from it
        supabase.table("nnn_lookup").select("id").limit(1).execute()
        logger.info("Table 'nnn_lookup' already exists")
        return True
    except Exception as e:
        logger.info(f"Table doesn't exist, will need manual creation: {e}")
        
        create_sql = '''
        CREATE TABLE nnn_lookup (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            diagnosis TEXT NOT NULL,
            definition TEXT,
            defining_characteristics TEXT[],
            related_factors TEXT[],
            risk_factors TEXT[],
            at_risk_population TEXT[],
            associated_conditions TEXT[],
            suggested_outcomes TEXT[],
            suggested_interventions TEXT[],
            embedding VECTOR(768),
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        -- Create indexes
        CREATE INDEX ON nnn_lookup USING ivfflat (embedding vector_cosine_ops);
        CREATE INDEX ON nnn_lookup USING GIN (at_risk_population);
        CREATE INDEX ON nnn_lookup USING GIN (associated_conditions);
        CREATE INDEX idx_nnn_lookup_diagnosis ON nnn_lookup (diagnosis);
        '''
        
        logger.error("FAILED: Cannot create table automatically via script.")
        logger.info("Please create the table manually in Supabase SQL editor:")
        logger.info("="*60)
        logger.info(create_sql)
        logger.info("="*60)
        logger.info("After creating the table, run this script again.")
        return False

def main():
    """Main migration function."""
    logger.info("Starting NNN content migration to Supabase...")
    
    # Create table if it doesn't exist
    if not create_table_if_not_exists():
        logger.error("FAILED: Table does not exist. Please create manually and run again.")
        return
    
    # Load the normalized content
    try:
        with open("normalized_NNN_content.json", "r", encoding="utf-8") as f:
            all_entries = json.load(f)
        
        # Filter out empty entries and entries without diagnosis
        entries = [entry for entry in all_entries 
                  if entry and entry.get("diagnosis") and entry.get("diagnosis").strip()]
        
        logger.info(f"Loaded {len(all_entries)} total entries, filtered to {len(entries)} valid entries")
        
    except FileNotFoundError:
        logger.error("normalized_NNN_content.json not found!")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in normalized_NNN_content.json: {e}")
        return
    
    success_count = 0
    skipped_count = 0
    failed_count = 0
    entries_with_embeddings = []
    
    # Process each entry
    for i, entry in enumerate(tqdm(entries, desc="Migrating NNN entries")):
        diagnosis = entry.get("diagnosis", f"unknown_{i}")
        
        try:
            # Check if entry already exists (for idempotency)
            if check_existing_entry(diagnosis):
                logger.info(f"Skipping existing entry: {diagnosis}")
                skipped_count += 1
                continue
            
            # Generate embedding text
            embedding_text = make_embedding_text(entry)
            if not embedding_text.strip():
                logger.warning(f"Empty embedding text for: {diagnosis}")
                failed_count += 1
                continue
            
            # Generate embedding
            logger.debug(f"Processing: {diagnosis}")
            embedding = get_embedding(embedding_text)
            
            # Insert into database
            if insert_row(entry, embedding):
                success_count += 1
                
                # Add to backup data
                entry_with_embedding = entry.copy()
                entry_with_embedding["embedding"] = embedding
                entry_with_embedding["embedding_text"] = embedding_text
                entries_with_embeddings.append(entry_with_embedding)
            else:
                failed_count += 1
                
        except Exception as e:
            logger.error(f"Failed to process entry '{diagnosis}': {e}")
            failed_count += 1
    
    # Save backup file with embeddings
    if entries_with_embeddings:
        save_backup_with_embeddings(entries_with_embeddings)
    
    # Final report
    total_processed = success_count + skipped_count + failed_count
    logger.info("=" * 50)
    logger.info("MIGRATION COMPLETE")
    logger.info("=" * 50)
    logger.info(f"Total entries processed: {total_processed}")
    logger.info(f"Successfully migrated: {success_count}")
    logger.info(f"Skipped (already exists): {skipped_count}")
    logger.info(f"Failed: {failed_count}")
    logger.info(f"Success rate: {(success_count / len(entries) * 100):.1f}%")
    
    if failed_count > 0:
        logger.warning(f"Check migration.log for details about {failed_count} failed entries")

if __name__ == "__main__":
    main()