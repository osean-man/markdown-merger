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
		Use:   "md-merge",
		Short: "md-merge merges all markdown files into one",
		Run: func(cmd *cobra.Command, args []string) {
			fmt.Println("Merging Markdown files...")
		},
	}

	var path string
	var output string

	rootCmd.Flags().StringVarP(&path, "path", "p", ".", "Path to recursively merge Markdown files from")
	rootCmd.Flags().StringVarP(&output, "output", "o", "merged.md", "Output file name")

	rootCmd.MarkFlagRequired("path")

	rootCmd.Run = func(cmd *cobra.Command, args []string) {
		mergeMarkdownFiles(path, output)
	}

	if err := rootCmd.Execute(); err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
}

func mergeMarkdownFiles(path, outputFileName string) {
	var mergedContent strings.Builder

	err := filepath.Walk(path, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && strings.HasSuffix(info.Name(), ".md") {
			content, err := ioutil.ReadFile(path)
			if err != nil {
				return err
			}
			mergedContent.WriteString(string(content) + "\n\n")
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

	fmt.Printf("Markdown files have been merged into %s\n", outputFileName)
}

