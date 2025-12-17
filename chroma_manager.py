import chromadb

DB_PATH = "./chroma_db"
COLLECTION_NAME = "tabular_data"

client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


# -------------------------------
# VIEW / OPEN FUNCTIONS
# -------------------------------

def show_titles():
    """
    Show unique source titles (e.g., CSV / Excel file names)
    """
    results = collection.get(include=["metadatas"])

    if not results["metadatas"]:
        print("\nNo data found in the collection.\n")
        return

    sources = sorted(
        set(m.get("source", "unknown") for m in results["metadatas"])
    )

    print("\nAvailable data sources:\n")
    for i, src in enumerate(sources, 1):
        print(f"{i}. {src}")

    print(f"\nTotal sources: {len(sources)}\n")


def show_row_count_per_source():
    """
    Show how many rows each source has
    """
    results = collection.get(include=["metadatas"])

    if not results["metadatas"]:
        print("\nNo data found.\n")
        return

    counts = {}
    for m in results["metadatas"]:
        src = m.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1

    print("\nRow count per source:\n")
    for src, count in counts.items():
        print(f"{src}: {count} rows")
    print()


def preview_source():
    """
    Preview a few embedded rows from a selected source
    """
    source = input("Enter source name to preview (e.g., pokemon.csv): ").strip()

    results = collection.get(
        where={"source": source},
        include=["documents"]
    )

    if not results["documents"]:
        print("No rows found for this source.\n")
        return

    print(f"\nPreviewing first 5 rows from {source}:\n")
    for i, doc in enumerate(results["documents"][:5], 1):
        print(f"{i}. {doc}\n")


# -------------------------------
# CLEAR FUNCTIONS
# -------------------------------

def clear_all():
    confirm = input(
        "⚠ This will DELETE ALL data in the collection. Type YES to continue: "
    )
    if confirm == "YES":
        client.delete_collection(COLLECTION_NAME)
        print("✓ Collection deleted completely.\n")
    else:
        print("Cancelled.\n")


def clear_by_source():
    source = input("Enter source file name (e.g., pokemon.csv): ").strip()

    result = collection.get(
        where={"source": source}
        # DO NOT include ["ids"]
    )

    ids = result.get("ids", [])

    if not ids:
        print("No records found for that source.\n")
        return

    confirm = input(
        f"⚠ Delete {len(ids)} rows from {source}? Type YES to continue: "
    )

    if confirm == "YES":
        collection.delete(ids=ids)
        print(f"✓ Deleted rows from {source}\n")
    else:
        print("Cancelled.\n")


def clear_by_metadata():
    key = input("Metadata key (e.g., type1, legendary): ").strip()
    value = input("Metadata value (e.g., Fire, True): ").strip()

    result = collection.get(
        where={key: value},
        include=["ids"]
    )

    if not result["ids"]:
        print("No matching records found.\n")
        return

    confirm = input(
        f"⚠ Delete {len(result['ids'])} rows where {key}={value}? Type YES to continue: "
    )

    if confirm == "YES":
        collection.delete(ids=result["ids"])
        print("✓ Matching rows deleted.\n")
    else:
        print("Cancelled.\n")


# -------------------------------
# MENU
# -------------------------------

def menu():
    print("""
Chroma DB Manager

1. Show available titles (sources)
2. Show row count per source
3. Preview rows from a source
4. Clear entire collection
5. Clear data by source file
6. Clear data by metadata key/value
7. Exit
""")


if __name__ == "__main__":
    while True:
        menu()
        choice = input("Enter choice: ").strip()

        if choice == "1":
            show_titles()
        elif choice == "2":
            show_row_count_per_source()
        elif choice == "3":
            preview_source()
        elif choice == "4":
            clear_all()
        elif choice == "5":
            clear_by_source()
        elif choice == "6":
            clear_by_metadata()
        elif choice == "7":
            print("Bye.")
            break
        else:
            print("Invalid choice.\n")
