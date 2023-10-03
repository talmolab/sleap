import json
import os

# Set the file paths
input_md_file = os.path.join(os.path.dirname(__file__), 'bulletin.md')
output_json_file = os.path.join(os.path.dirname(__file__), 'bulletin.json')

def generate_json_file():
    with open(input_md_file, 'r', encoding='utf-8') as md_file:
        markdown_content = md_file.read()
    bulletin_json = []
    content = ''

    # Initialize title and date with default values
    title = "DEFAULT_TITLE"
    date = "DEFAULT_DATE"

    for line in markdown_content.split('\n'):
        if line.startswith('---'):
            bulletin_json.append({'title': title, 'date': date, 'content':content})
            content = ''
            # Reset title and date to their default values after each section
            title = "DEFAULT_TITLE"
            date = "DEFAULT_DATE"
        elif line.startswith('##'):
            title = line[3:].strip()
        elif line.startswith('_'):
            date = line[1:len(line)-1].strip()
        else:
            content += (line + '\n')
    # Append last section   
    bulletin_json.append({'title': title, 'date': date, 'content':content})
    
    with open(output_json_file, 'w', encoding='utf-8') as json_file:
        json.dump(bulletin_json, json_file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    generate_json_file()
