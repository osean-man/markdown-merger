package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"
)

func main() {
	var rootCmd = &cobra.Command{
		Use:   "file-merge",
		Short: "file-merge merges all files of a specified type into one",
		Run:   func(cmd *cobra.Command, args []string) {},
	}

	var path, output, extension string
	var separator bool

	rootCmd.Flags().StringVarP(&path, "path", "p", ".", "Path to recursively merge files from")
	rootCmd.Flags().StringVarP(&output, "output", "o", "merged.txt", "Output file name")
	rootCmd.Flags().StringVarP(&extension, "extension", "x", "txt", "File extension to merge")
	rootCmd.Flags().BoolVarP(&separator, "separator", "s", false, "Insert a separator comment between files")

	rootCmd.MarkFlagRequired("path")

	rootCmd.Run = func(cmd *cobra.Command, args []string) {
		mergeFiles(path, output, extension, separator)
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func mergeFiles(path, outputFileName, extension string, separator bool) {
	var mergedContent strings.Builder
	fileExt := "." + extension
	codeFence := getCodeFence(extension)

	err := filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), fileExt) {
			if separator {
				mergedContent.WriteString(fmt.Sprintf("// # %s Contents:\n\n", info.Name()))
			}
			if codeFence != "" {
				mergedContent.WriteString(codeFence + "\n")
			}
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			mergedContent.WriteString(string(content) + "\n")
			if codeFence != "" {
				mergedContent.WriteString("```\n") 
			}
			mergedContent.WriteString("\n")
		}
		return nil
	})

	if err != nil {
		fmt.Printf("An error occurred while merging files: %s\n", err)
		return
	}

	err = ioutil.WriteFile(outputFileName, []byte(mergedContent.String()), 0644)
	if err != nil {
		fmt.Printf("An error occurred while writing to file: %s\n", err)
		return
	}

	fmt.Printf("Files have been merged into %s\n", outputFileName)
}

func getCodeFence(extension string) string {
	switch extension {
	case "py":
		return "```python"
	case "go":
		return "```go"
	case "js":
		return "```javascript"
	case "java":
		return "```java"
	case "cs":
		return "```csharp"
	case "cpp", "cxx", "cc":
		return "```cpp"
	case "rb":
		return "```ruby"
	case "php":
		return "```php"
	default:
		return ""
	}
}
