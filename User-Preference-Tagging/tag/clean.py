import os
import csv
import re
from pathlib import Path
from datetime import datetime

# ANSI color codes for elegant output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header():
    """Print elegant header."""
    print(f"{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    📁 CSV CLEANER TOOL                       ║")
    print("║                Remove duplicates & organize data             ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")

def print_progress_bar(current, total, width=50):
    """Print a beautiful progress bar."""
    progress = int(width * current / total)
    bar = "█" * progress + "░" * (width - progress)
    percentage = (current / total) * 100
    print(f"\r{Colors.CYAN}Progress: [{bar}] {percentage:.1f}% ({current}/{total}){Colors.END}", end="", flush=True)

def clean_csv_file(file_path, file_num, total_files):
    """Clean a single CSV file by removing duplicates and sorting entries."""
    filename = file_path.name
    
    # Print file header
    print(f"\n{Colors.BOLD}{Colors.BLUE}📄 Processing File {file_num}/{total_files}: {filename}{Colors.END}")
    print(f"{Colors.CYAN}{'─' * 60}{Colors.END}")
    
    # Read all entries from the CSV file
    entries = []
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file)
            for row in reader:
                if row and row[0].strip():  # Skip empty rows
                    entry = row[0].strip()
                    entries.append(entry)
    except FileNotFoundError:
        print(f"{Colors.RED}  ❌ Warning: File {filename} not found, skipping...{Colors.END}")
        return
    except Exception as e:
        print(f"{Colors.RED}  ❌ Error reading {filename}: {e}{Colors.END}")
        return
    
    if not entries:
        print(f"{Colors.YELLOW}  ⚠️  Warning: No entries found in {filename}{Colors.END}")
        return
    
    # Find and show duplicates
    seen = {}
    duplicates_found = []
    
    for i, entry in enumerate(entries):
        entry_lower = entry.lower()
        if entry_lower in seen:
            # This is a duplicate
            original_index = seen[entry_lower]
            original_entry = entries[original_index]
            duplicates_found.append({
                'original': original_entry,
                'duplicate': entry,
                'original_line': original_index + 1,
                'duplicate_line': i + 1
            })
        else:
            seen[entry_lower] = i
    
    # Show duplicates found with elegant formatting
    if duplicates_found:
        print(f"{Colors.YELLOW}  🔍 Found {len(duplicates_found)} duplicate(s):{Colors.END}")
        print()
        for i, dup in enumerate(duplicates_found, 1):
            print(f"    {Colors.CYAN}{i}.{Colors.END} Duplicate Entry:")
            print(f"       {Colors.GREEN}✓ Keep:   {Colors.END}Line {dup['original_line']:3d} → '{dup['original']}'")
            print(f"       {Colors.RED}✗ Remove: {Colors.END}Line {dup['duplicate_line']:3d} → '{dup['duplicate']}'")
            print()
    else:
        print(f"{Colors.GREEN}  ✅ No duplicates found{Colors.END}")
    
    # Remove duplicates while preserving order and capitalization
    seen = set()
    unique_entries = []
    for entry in entries:
        # Use lowercase for comparison to catch duplicates with different capitalization
        entry_lower = entry.lower()
        if entry_lower not in seen:
            seen.add(entry_lower)
            unique_entries.append(entry)
    
    # Sort entries alphabetically (case-insensitive)
    unique_entries.sort(key=str.lower)
    
    # Write back to the file
    try:
        with open(file_path, 'w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file)
            for entry in unique_entries:
                writer.writerow([entry])
        
        removed_count = len(entries) - len(unique_entries)
        
        # Print results with elegant formatting
        print(f"{Colors.BOLD}📊 Results:{Colors.END}")
        print(f"   {Colors.CYAN}• Total entries: {Colors.END}{len(entries):4d}")
        print(f"   {Colors.GREEN}• Unique entries: {Colors.END}{len(unique_entries):4d}")
        if removed_count > 0:
            print(f"   {Colors.RED}• Duplicates removed: {Colors.END}{removed_count:4d}")
            print(f"   {Colors.YELLOW}• Space saved: {Colors.END}{removed_count} entries")
        else:
            print(f"   {Colors.GREEN}• No duplicates found{Colors.END}")
            
    except Exception as e:
        print(f"{Colors.RED}  ❌ Error writing {filename}: {e}{Colors.END}")

def print_summary(stats):
    """Print elegant summary."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                        📈 SUMMARY                            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    total_files = stats['total_files']
    processed_files = stats['processed_files']
    total_entries = stats['total_entries']
    total_unique = stats['total_unique']
    total_duplicates = stats['total_duplicates']
    
    print(f"{Colors.BOLD}📁 Files Processed:{Colors.END}")
    print(f"   {Colors.CYAN}• Total files found: {Colors.END}{total_files}")
    print(f"   {Colors.GREEN}• Successfully processed: {Colors.END}{processed_files}")
    
    if total_files != processed_files:
        print(f"   {Colors.RED}• Failed to process: {Colors.END}{total_files - processed_files}")
    
    print(f"\n{Colors.BOLD}📊 Data Statistics:{Colors.END}")
    print(f"   {Colors.CYAN}• Total entries processed: {Colors.END}{total_entries:,}")
    print(f"   {Colors.GREEN}• Unique entries kept: {Colors.END}{total_unique:,}")
    print(f"   {Colors.RED}• Duplicates removed: {Colors.END}{total_duplicates:,}")
    
    if total_duplicates > 0:
        space_saved = (total_duplicates / total_entries) * 100
        print(f"   {Colors.YELLOW}• Space saved: {Colors.END}{space_saved:.1f}%")
    
    print(f"\n{Colors.GREEN}{Colors.BOLD}✨ All CSV files have been cleaned and refreshed!{Colors.END}")
    print(f"{Colors.CYAN}🕒 Completed at: {Colors.END}{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """Main function to clean all CSV files in the tags directory."""
    print_header()
    
    # Get the tags directory path
    tags_dir = Path("tags")
    
    if not tags_dir.exists():
        print(f"{Colors.RED}❌ Error: 'tags' directory not found!{Colors.END}")
        return
    
    # Find all CSV files in the tags directory
    csv_files = list(tags_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"{Colors.YELLOW}⚠️  No CSV files found in the tags directory!{Colors.END}")
        return
    
    print(f"{Colors.BOLD}📂 Found {len(csv_files)} CSV file(s) to process:{Colors.END}")
    for i, csv_file in enumerate(csv_files, 1):
        print(f"   {Colors.CYAN}{i:2d}.{Colors.END} {csv_file.name}")
    print()
    
    # Process each CSV file
    total_files = len(csv_files)
    processed_files = 0
    total_entries = 0
    total_unique = 0
    total_duplicates = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        # Get file stats before processing
        try:
            with open(csv_file, 'r', encoding='utf-8', newline='') as file:
                reader = csv.reader(file)
                file_entries = sum(1 for row in reader if row and row[0].strip())
                total_entries += file_entries
        except:
            file_entries = 0
        
        clean_csv_file(csv_file, i, total_files)
        processed_files += 1
        
        # Update progress
        total_unique += file_entries  # This will be updated after actual processing
        print_progress_bar(i, total_files)
    
    print()  # New line after progress bar
    
    # Print summary
    stats = {
        'total_files': total_files,
        'processed_files': processed_files,
        'total_entries': total_entries,
        'total_unique': total_unique,
        'total_duplicates': total_entries - total_unique
    }
    
    print_summary(stats)

if __name__ == "__main__":
    main() 
