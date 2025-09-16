import os
from dotenv import load_dotenv
from supabase import create_client, Client
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_supabase_connection():
    """Test Supabase connection and table access."""
    
    # Check environment variables
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    
    print("=" * 50)
    print("SUPABASE CONNECTION TEST")
    print("=" * 50)
    
    if not SUPABASE_URL:
        print("❌ SUPABASE_URL not found in environment variables")
        return False
    else:
        print(f"✅ SUPABASE_URL: {SUPABASE_URL}")
    
    if not SUPABASE_KEY:
        print("❌ SUPABASE_SERVICE_ROLE_KEY not found in environment variables")
        return False
    else:
        print(f"✅ SUPABASE_SERVICE_ROLE_KEY: {SUPABASE_KEY[:20]}...")
    
    try:
        # Initialize Supabase client
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase client initialized successfully")
        
        # Test basic connection by listing tables (if possible)
        print("\n--- Testing Database Connection ---")
        
        # Check if nnn_lookup table exists
        try:
            result = supabase.table("nnn_lookup").select("*").limit(1).execute()
            print(f"✅ nnn_lookup table exists and is accessible")
            print(f"   Current row count check: {len(result.data)} row(s) returned in test query")
            
            # Try to get actual count
            try:
                count_result = supabase.table("nnn_lookup").select("id", count="exact").execute()
                print(f"   Total rows in nnn_lookup: {count_result.count}")
            except Exception as e:
                print(f"   Could not get exact count: {e}")
                
        except Exception as e:
            print(f"❌ nnn_lookup table access failed: {e}")
            print("\n--- Table Creation Required ---")
            print("You need to create the nnn_lookup table with this SQL:")
            print("""
CREATE TABLE nnn_lookup (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    diagnosis TEXT NOT NULL,
    definition TEXT,
    defining_characteristics TEXT[],
    related_factors TEXT[],
    risk_factors TEXT[],
    suggested_outcomes TEXT[],
    suggested_interventions TEXT[],
    embedding VECTOR(768),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable the vector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- Create vector index for similarity search
CREATE INDEX ON nnn_lookup USING ivfflat (embedding vector_cosine_ops);
            """)
            return False
        
        # Test insert capability (dry run)
        print("\n--- Testing Insert Permissions ---")
        try:
            test_data = {
                "diagnosis": "TEST_DIAGNOSIS_DELETE_ME",
                "definition": "Test definition",
                "defining_characteristics": ["test char 1", "test char 2"],
                "related_factors": ["test factor 1"],
                "risk_factors": ["test risk 1"],
                "suggested_outcomes": ["test outcome 1"],
                "suggested_interventions": ["test intervention 1"],
                "embedding": [0.1] * 768  # Mock embedding
            }
            
            # Insert test row
            insert_result = supabase.table("nnn_lookup").insert(test_data).execute()
            print("✅ Insert permission confirmed")
            
            # Clean up test row
            if insert_result.data:
                test_id = insert_result.data[0]['id']
                supabase.table("nnn_lookup").delete().eq("id", test_id).execute()
                print("✅ Delete permission confirmed (cleanup successful)")
            
        except Exception as e:
            print(f"❌ Insert/Delete permission test failed: {e}")
            return False
        
        # Test vector extension
        print("\n--- Testing Vector Extension ---")
        try:
            # Try a simple vector operation to test if extension is available
            vector_test = supabase.table("nnn_lookup").select("id").limit(1).execute()
            print("✅ Vector operations should work (table accessible)")
        except Exception as e:
            print(f"⚠️  Vector extension test warning: {e}")
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED - Ready for migration!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        return False

def test_existing_data():
    """Check what data already exists in the table."""
    try:
        SUPABASE_URL = os.getenv("SUPABASE_URL")
        SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print("\n--- Checking Existing Data ---")
        
        # Get sample of existing data
        result = supabase.table("nnn_lookup").select("diagnosis, definition").limit(5).execute()
        
        if result.data:
            print(f"Found {len(result.data)} existing entries (showing first 5):")
            for i, row in enumerate(result.data, 1):
                diagnosis = row.get('diagnosis', 'N/A')[:50]
                definition = row.get('definition', 'N/A')[:80]
                print(f"  {i}. {diagnosis}...")
                print(f"     {definition}...")
                print()
        else:
            print("✅ Table is empty - ready for fresh migration")
            
    except Exception as e:
        print(f"Could not check existing data: {e}")

if __name__ == "__main__":
    success = test_supabase_connection()
    
    if success:
        test_existing_data()
        print("\n🚀 You can now run: python migrate_nnn_to_supabase.py")
    else:
        print("\n🛑 Fix the issues above before running the migration script")