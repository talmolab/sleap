import json

# Set the file paths
input_md_file = 'D:\TalmoLab\sleap\docs\bulletin.md'
output_json_file = 'D:\TalmoLab\sleap\docs\\bulletin.json'

def generate_json_file():
    with open(input_md_file, 'r', encoding='utf-8') as md_file:
        markdown_content = md_file.read()
    bulletin_json = []
    content = ''

    for line in markdown_content.split('\n'):
        if line.startswith('---'):
            bulletin_json.append({'title': title, 'date': date, 'content':content})
            content = ''
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
