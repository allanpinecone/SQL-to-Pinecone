import sqlite3
import os
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load keys
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

index_name = "employee-demo"

# Create Pinecone index if not exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # OpenAI embeddings size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# ---- Step 1: Create SQL dataset ----
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

cur.execute("""
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    job_title TEXT,
    department TEXT,
    bio TEXT
)
""")

sample_data = [
    (1, "Alice Johnson", "Data Scientist", "Finance", "Works on financial risk models."),
    (2, "Bob Smith", "Software Engineer", "IT", "Backend systems specialist."),
    (3, "Carol Lee", "HR Manager", "Human Resources", "Focuses on employee well-being."),
    (4, "David Kim", "Product Manager", "Marketing", "Leads product campaigns."),
]

cur.executemany("INSERT INTO employees VALUES (?, ?, ?, ?, ?)", sample_data)
conn.commit()

# ---- Step 2: Push data into Pinecone ----
rows = cur.execute("SELECT id, name, job_title, department, bio FROM employees").fetchall()

vectors = []
for row in rows:
    emp_id, name, job, dept, bio = row
    text = f"{name}, {job} in {dept}. {bio}"
    emb = openai_client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
    vectors.append({"id": str(emp_id), "values": emb, "metadata": {"name": name, "job": job, "dept": dept, "bio": bio}})

index.upsert(vectors=vectors)

print("✅ Data uploaded to Pinecone!")

# ---- Step 3: Query Pinecone ----
def search(query: str):
    emb = openai_client.embeddings.create(input=query, model="text-embedding-3-small").data[0].embedding
    results = index.query(vector=emb, top_k=3, include_metadata=True)
    print(f"\n🔍 Query: {query}")
    for match in results.matches:
        md = match.metadata
        print(f"- {md['name']} ({md['job']} in {md['dept']}) | Score: {match.score:.2f}")

# Demo queries
search("Who works in finance?")
search("Show me someone in HR")
