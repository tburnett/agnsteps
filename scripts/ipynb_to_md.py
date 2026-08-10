import nbformat

# Load the notebook
nb = nbformat.read("my_notebook.ipynb", as_version=4)

md_lines = []

for cell in nb.cells:
    if cell.cell_type == "markdown":
        # Keep markdown content as-is
        md_lines.append(cell.source)
    elif cell.cell_type == "code":
        # Keep only outputs
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                md_lines.append(f"```\n{output.text.strip()}\n```")
            elif output.output_type == "execute_result" and "text/plain" in output.data:
                md_lines.append(f"```\n{output.data['text/plain'].strip()}\n```")
            elif output.output_type == "display_data" and "image/png" in output.data:
                img_data = output.data["image/png"]
                md_lines.append(f"![output](data:image/png;base64,{img_data})")

# Save to Markdown file
with open("report.md", "w", encoding="utf-8") as f:
    f.write("\n\n".join(md_lines))

print("✅ Export complete: report.md created")
