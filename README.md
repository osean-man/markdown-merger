# Markdown Merger (mdmerge)

`mdmerge` is a command-line interface (CLI) tool built with Go and Cobra, designed to facilitate the merging of multiple Markdown files into a single document. This tool was created to quickly aggregate documentation from git repositories, making it easier to feed content into Large Language Models (LLMs) for processing or analysis. While currently focused on Markdown files, `mdmerge` can be extended to support various file types.

## UPDATED TO USE ANY FILE TYPE AND EXTRA COMMANDS

## Features

- **Recursive Merging**: Traverse directories recursively to find and merge all Markdown files.
- **Customizable Output**: Specify the output file name for the merged content.
- **Ease of Use**: Simple and straightforward CLI commands.

## Installation

To install `mdmerge`, clone the repository and build the project:

```bash
git clone https://github.com/osean-man/markdown-merger.git
cd mdmerge
go build -o mdmerge
chmod +x mdmerge
```
Usage
After building the tool, run it using the following syntax:
```bash
./mdmerge --path=<path-to-markdown-files> --output=<merged-filename.md>
```
Replace <path-to-markdown-files> with the directory path where your Markdown files are located, and <merged-filename.md> with your desired output file name.
## Flags
```
--path, -p: (Required) The path to the directory containing Markdown files to be merged.
--output, -o: (Optional) The name of the output file. Defaults to merged.md if not specified.
```
## Examples
Merge Markdown files from a specific directory into a file named combined.md:
```bash
./mdmerge --path=/path/to/docs --output=combined.md
```
## Extending to Other File Types
The simple architecture of the mdmerge script allows for easy extension to support any additional file types. Submit a pull request if you think that is useful.

## License
[MIT](https://choosealicense.com/licenses/mit/)
