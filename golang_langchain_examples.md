// # anthropic_completion_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/anthropic"
)

func main() {
	llm, err := anthropic.New()
	// note: You would include anthropic.WithModel("claude-2") to use the claude-2 model.
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, "Human: Who was the first man to walk on the moon?\nAssistant:",
		llms.WithTemperature(0.8),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	_ = completion
}

```

// # chroma_vectorstore_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	chroma_go "github.com/amikos-tech/chroma-go"
	"github.com/google/uuid"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

func main() {

	// Create a new Chroma vector store.
	store, errNs := chroma.New(
		chroma.WithChromaURL(os.Getenv("CHROMA_URL")),
		chroma.WithOpenAiAPIKey(os.Getenv("OPENAI_API_KEY")),
		chroma.WithDistanceFunction(chroma_go.COSINE),
		chroma.WithNameSpace(uuid.New().String()),
	)
	if errNs != nil {
		log.Fatalf("new: %v\n", errNs)
	}

	type meta = map[string]any

	// Add documents to the vector store.
	errAd := store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "Tokyo", Metadata: meta{"population": 9.7, "area": 622}},
		{PageContent: "Kyoto", Metadata: meta{"population": 1.46, "area": 828}},
		{PageContent: "Hiroshima", Metadata: meta{"population": 1.2, "area": 905}},
		{PageContent: "Kazuno", Metadata: meta{"population": 0.04, "area": 707}},
		{PageContent: "Nagoya", Metadata: meta{"population": 2.3, "area": 326}},
		{PageContent: "Toyota", Metadata: meta{"population": 0.42, "area": 918}},
		{PageContent: "Fukuoka", Metadata: meta{"population": 1.59, "area": 341}},
		{PageContent: "Paris", Metadata: meta{"population": 11, "area": 105}},
		{PageContent: "London", Metadata: meta{"population": 9.5, "area": 1572}},
		{PageContent: "Santiago", Metadata: meta{"population": 6.9, "area": 641}},
		{PageContent: "Buenos Aires", Metadata: meta{"population": 15.5, "area": 203}},
		{PageContent: "Rio de Janeiro", Metadata: meta{"population": 13.7, "area": 1200}},
		{PageContent: "Sao Paulo", Metadata: meta{"population": 22.6, "area": 1523}},
	})
	if errAd != nil {
		log.Fatalf("AddDocument: %v\n", errAd)
	}

	ctx := context.TODO()

	type exampleCase struct {
		name         string
		query        string
		numDocuments int
		options      []vectorstores.Option
	}

	type filter = map[string]any

	exampleCases := []exampleCase{
		{
			name:         "Up to 5 Cities in Japan",
			query:        "Which of these are cities are located in Japan?",
			numDocuments: 5,
			options: []vectorstores.Option{
				vectorstores.WithScoreThreshold(0.8),
			},
		},
		{
			name:         "A City in South America",
			query:        "Which of these are cities are located in South America?",
			numDocuments: 1,
			options: []vectorstores.Option{
				vectorstores.WithScoreThreshold(0.8),
			},
		},
		{
			name:         "Large Cities in South America",
			query:        "Which of these are cities are located in South America?",
			numDocuments: 100,
			options: []vectorstores.Option{
				vectorstores.WithFilters(filter{
					"$and": []filter{
						{"area": filter{"$gte": 1000}},
						{"population": filter{"$gte": 13}},
					},
				}),
			},
		},
	}

	// run the example cases
	results := make([][]schema.Document, len(exampleCases))
	for ecI, ec := range exampleCases {
		docs, errSs := store.SimilaritySearch(ctx, ec.query, ec.numDocuments, ec.options...)
		if errSs != nil {
			log.Fatalf("query1: %v\n", errSs)
		}
		results[ecI] = docs
	}

	// print out the results of the run
	fmt.Printf("Results:\n")
	for ecI, ec := range exampleCases {
		texts := make([]string, len(results[ecI]))
		for docI, doc := range results[ecI] {
			texts[docI] = doc.PageContent
		}
		fmt.Printf("%d. case: %s\n", ecI+1, ec.name)
		fmt.Printf("    result: %s\n", strings.Join(texts, ", "))
	}

}

```

// # cohere_completion_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms/cohere"
)

func main() {
	llm, err := cohere.New()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	input := "The first man to walk on the moon"
	completion, err := llm.Call(ctx, input)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)

	inputToken := llm.GetNumTokens(input)
	outputToken := llm.GetNumTokens(completion)

	fmt.Printf("%v/%v\n", inputToken, outputToken)
}

```

// # document_qa.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	llm, err := openai.New()
	if err != nil {
		return err
	}

	// We can use LoadStuffQA to create a chain that takes input documents and a question,
	// stuffs all the documents into the prompt of the llm and returns an answer to the
	// question. It is suitable for a small number of documents.
	stuffQAChain := chains.LoadStuffQA(llm)
	docs := []schema.Document{
		{PageContent: "Harrison went to Harvard."},
		{PageContent: "Ankush went to Princeton."},
	}

	answer, err := chains.Call(context.Background(), stuffQAChain, map[string]any{
		"input_documents": docs,
		"question":        "Where did Harrison go to collage?",
	})
	if err != nil {
		return err
	}
	fmt.Println(answer)

	// Another option is to use the refine documents chain for question answering. This
	// chain iterates over the input documents one by one, updating an intermediate answer
	// with each iteration. It uses the previous version of the answer and the next document
	// as context. The downside of this type of chain is that it uses multiple llm calls that
	// cant be done in parallel.
	refineQAChain := chains.LoadRefineQA(llm)
	answer, err = chains.Call(context.Background(), refineQAChain, map[string]any{
		"input_documents": docs,
		"question":        "Where did Ankush go to collage?",
	})
	fmt.Println(answer)

	return nil
}

```

// # ernie_completion_example.go Contents:

```go
package main

func main() {
	//llm, err := ernie.New(ernie.WithModelName(ernie.ModelNameERNIEBot))
	// note:
	// You would include ernie.WithAKSK(apiKey,secretKey) to use specific auth info.
	// You would include ernie.WithModelName(ernie.ModelNameERNIEBot) to use the ERNIE-Bot model.
	//if err != nil {
	//log.Fatal(err)
	//}
	//ctx := context.Background()
	//completion, err := llm.Call(ctx, "介绍一下你自己",
	//llms.WithTemperature(0.8),
	//llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
	//log.Println(string(chunk))
	//return nil
	//}),
	//)

	//if err != nil {
	//log.Fatal(err)
	//}

	//_ = completion

	//// embedding
	//embedding, _ := ernieembedding.NewErnie()

	//emb, err := embedding.EmbedDocuments(ctx, []string{"你好"})

	//if err != nil {
	//log.Fatal(err)
	//}
	//fmt.Println("Embedding-V1:", len(emb), len(emb[0]))
}

```

// # huggingface_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/huggingface"
)

func main() {
	// You may instantiate a client with a custom token and/or custom model
	// clientOptions := []huggingface.Option{
	// 	huggingface.WithToken("HF_1234"),
	// 	huggingface.WithModel("ZZZ"),
	// }
	// llm, err := huggingface.New(clientOptions...)

	// Or you may instantiate a client with a default model and use token from environment variable
	llm, err := huggingface.New()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()

	// By default, library will use default model described in huggingface.defaultModel
	// completion, err := llm.Call(ctx, "What would be a good company name be for name a company that makes colorful socks?")

	// Or override default model to another one
	generateOptions := []llms.CallOption{
		llms.WithModel("gpt2"),
		// llms.WithTopK(10),
		// llms.WithTopP(0.95),
		// llms.WithSeed(13),
	}
	completion, err := llm.Call(ctx, "What would be a good company name be for name a company that makes colorful socks?", generateOptions...)
	// Check for errors
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(completion)
}

```

// # llm_chain.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	// We can construct an LLMChain from a PromptTemplate and an LLM.
	llm, err := openai.New()
	if err != nil {
		return err
	}
	prompt := prompts.NewPromptTemplate(
		"What is a good name for a company that makes {{.product}}?",
		[]string{"product"},
	)
	llmChain := chains.NewLLMChain(llm, prompt)

	// If a chain only needs one input we can use the run function to execute chain.
	ctx := context.Background()
	out, err := chains.Run(ctx, llmChain, "socks")
	if err != nil {
		return err
	}
	fmt.Println(out)

	translatePrompt := prompts.NewPromptTemplate(
		"Translate the following text from {{.inputLanguage}} to {{.outputLanguage}}. {{.text}}",
		[]string{"inputLanguage", "outputLanguage", "text"},
	)
	llmChain = chains.NewLLMChain(llm, translatePrompt)

	// Otherwise the call function must be used.
	outputValues, err := chains.Call(ctx, llmChain, map[string]any{
		"inputLanguage":  "English",
		"outputLanguage": "French",
		"text":           "I love programming.",
	})
	if err != nil {
		return err
	}

	out, ok := outputValues[llmChain.OutputKey].(string)
	if !ok {
		return fmt.Errorf("invalid chain return")
	}
	fmt.Println(out)

	return nil
}

```

// # llm_math_chain.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	llm, err := openai.New()
	if err != nil {
		return err
	}
	llmMathChain := chains.NewLLMMathChain(llm)
	ctx := context.Background()
	out, err := chains.Run(ctx, llmMathChain, "What is 1024 raised to the 0.43 power?")
	fmt.Println(out)
	return err
}

```

// # llm_summarization_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/documentloaders"
	"github.com/tmc/langchaingo/llms/vertexai"
	"github.com/tmc/langchaingo/textsplitter"
)

func main() {
	llm, err := vertexai.New()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()
	llmSummarizationChain := chains.LoadRefineSummarization(llm)
	doc := `AI applications are summarizing articles, writing stories and 
	engaging in long conversations — and large language models are doing 
	the heavy lifting.
	
	A large language model, or LLM, is a deep learning model that can 
	understand, learn, summarize, translate, predict, and generate text and other 
	content based on knowledge gained from massive datasets.
	
	Large language models - successful applications of 
	transformer models. They aren’t just for teaching AIs human languages, 
	but for understanding proteins, writing software code, and much, much more.
	
	In addition to accelerating natural language processing applications — 
	like translation, chatbots, and AI assistants — large language models are 
	used in healthcare, software development, and use cases in many other fields.`
	docs, err := documentloaders.NewText(strings.NewReader(doc)).LoadAndSplit(ctx,
		textsplitter.NewRecursiveCharacter(),
	)
	outputValues, err := chains.Call(ctx, llmSummarizationChain, map[string]any{"input_documents": docs})
	if err != nil {
		log.Fatal(err)
	}
	out := outputValues["text"].(string)
	fmt.Println(out)

	// Output:
	// Large language models are a type of deep learning model that can understand, learn,
	// summarize, translate, predict, and generate text and other content based on knowledge
	// gained from massive datasets. They are used in a variety of applications, including
	// natural language processing, healthcare, and software development.
}

```

// # local_llm_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms/local"
)

func main() {
	// You may instantiate a client with a default bin and args from environment variable
	llm, err := local.New()
	if err != nil {
		log.Fatal(err)
	}

	// Or instantiate a client with a custom bin and args options
	//clientOptions := []local.Option{
	//	local.WithBin("/usr/bin/echo"),
	//	local.WithArgs("--arg1=value1 --arg2=value2"),
	//	local.WithGlobalAsArgs(), // build key-value arguments from global llms.Options, then append to args
	//}
	//llm, err := local.New(clientOptions...)

	// Init context
	ctx := context.Background()

	// By default, library will use default bin and args
	completion, err := llm.Call(ctx, "How many sides does a square have?")
	// Or append to default args options from global llms.Options
	//generateOptions := []llms.CallOption{
	//	llms.WithTopK(10),
	//	llms.WithTopP(0.95),
	//	llms.WithSeed(13),
	//}
	// In that case command will look like: /path/to/bin --arg1=value1 --arg2=value2 --top_k=10 --top_p=0.95 --seed=13 "How many sides does a square have?"
	//completion, err := llm.Call(ctx, "How many sides does a square have?", generateOptions...)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(completion)
}

```

// # mrkl_agent.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
	"github.com/tmc/langchaingo/tools/serpapi"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	llm, err := openai.New()
	if err != nil {
		return err
	}
	search, err := serpapi.New()
	if err != nil {
		return err
	}
	agentTools := []tools.Tool{
		tools.Calculator{},
		search,
	}
	executor, err := agents.Initialize(
		llm,
		agentTools,
		agents.ZeroShotReactDescription,
		agents.WithMaxIterations(3),
	)
	if err != nil {
		return err
	}
	question := "Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?"
	answer, err := chains.Run(context.Background(), executor, question)
	fmt.Println(answer)
	return err
}

```

// # ollama_chat_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	llm, err := ollama.NewChat(ollama.WithLLMOptions(ollama.WithModel("llama2")))
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, []schema.ChatMessage{
		schema.SystemChatMessage{Content: "Give a precise answer to the question based on the context. Don't be verbose."},
		schema.HumanChatMessage{Content: "What would be a good company name a company that makes colorful socks? Give me 3 examples."},
	}, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Print(string(chunk))
		return nil
	}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}

```

// # chroma_vectorstore_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	chroma_go "github.com/amikos-tech/chroma-go"
	"github.com/google/uuid"
	ollama_emb "github.com/tmc/langchaingo/embeddings/ollama"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/chroma"
)

func main() {

	// Create our ollama LLM to use as embedder
	// Please note that current LLM generate poor embeddings
	ollamaLLM, err := ollama.New(ollama.WithModel("llama2"))
	if err != nil {
		log.Fatal(err)
	}
	ollamaEmbeder, err := ollama_emb.NewOllama(ollama_emb.WithClient(*ollamaLLM))
	if err != nil {
		log.Fatal(err)
	}

	// Create a new Chroma vector store.
	store, errNs := chroma.New(
		chroma.WithChromaURL(os.Getenv("CHROMA_URL")),
		chroma.WithEmbedder(ollamaEmbeder),
		chroma.WithDistanceFunction(chroma_go.COSINE),
		chroma.WithNameSpace(uuid.New().String()),
	)
	if errNs != nil {
		log.Fatalf("new: %v\n", errNs)
	}

	type meta = map[string]any

	// Add documents to the vector store.
	errAd := store.AddDocuments(context.Background(), []schema.Document{
		{PageContent: "Tokyo", Metadata: meta{"population": 9.7, "area": 622}},
		{PageContent: "Kyoto", Metadata: meta{"population": 1.46, "area": 828}},
		{PageContent: "Hiroshima", Metadata: meta{"population": 1.2, "area": 905}},
		{PageContent: "Kazuno", Metadata: meta{"population": 0.04, "area": 707}},
		{PageContent: "Nagoya", Metadata: meta{"population": 2.3, "area": 326}},
		{PageContent: "Toyota", Metadata: meta{"population": 0.42, "area": 918}},
		{PageContent: "Fukuoka", Metadata: meta{"population": 1.59, "area": 341}},
		{PageContent: "Paris", Metadata: meta{"population": 11, "area": 105}},
		{PageContent: "London", Metadata: meta{"population": 9.5, "area": 1572}},
		{PageContent: "Santiago", Metadata: meta{"population": 6.9, "area": 641}},
		{PageContent: "Buenos Aires", Metadata: meta{"population": 15.5, "area": 203}},
		{PageContent: "Rio de Janeiro", Metadata: meta{"population": 13.7, "area": 1200}},
		{PageContent: "Sao Paulo", Metadata: meta{"population": 22.6, "area": 1523}},
	})
	if errAd != nil {
		log.Fatalf("AddDocument: %v\n", errAd)
	}

	ctx := context.TODO()

	type exampleCase struct {
		name         string
		query        string
		numDocuments int
		options      []vectorstores.Option
	}

	type filter = map[string]any

	exampleCases := []exampleCase{
		{
			name:         "Up to 5 Cities in Japan",
			query:        "Which of these are cities are located in Japan?",
			numDocuments: 5,
			options: []vectorstores.Option{
				vectorstores.WithScoreThreshold(0.8),
			},
		},
		{
			name:         "A City in South America",
			query:        "Which of these are cities are located in South America?",
			numDocuments: 1,
			options: []vectorstores.Option{
				vectorstores.WithScoreThreshold(0.8),
			},
		},
		{
			name:         "Large Cities in South America",
			query:        "Which of these are cities are located in South America?",
			numDocuments: 100,
			options: []vectorstores.Option{
				vectorstores.WithFilters(filter{
					"$and": []filter{
						{"area": filter{"$gte": 1000}},
						{"population": filter{"$gte": 13}},
					},
				}),
			},
		},
	}

	// run the example cases
	results := make([][]schema.Document, len(exampleCases))
	for ecI, ec := range exampleCases {
		docs, errSs := store.SimilaritySearch(ctx, ec.query, ec.numDocuments, ec.options...)
		if errSs != nil {
			log.Fatalf("query1: %v\n", errSs)
		}
		results[ecI] = docs
	}

	// print out the results of the run
	fmt.Printf("Results:\n")
	for ecI, ec := range exampleCases {
		texts := make([]string, len(results[ecI]))
		for docI, doc := range results[ecI] {
			texts[docI] = doc.PageContent
		}
		fmt.Printf("%d. case: %s\n", ecI+1, ec.name)
		fmt.Printf("    result: %s\n", strings.Join(texts, ", "))
	}

}

```

// # ollama_completion_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

func main() {
	llm, err := ollama.New(ollama.WithModel("llama2"))
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, "Human: Who was the first man to walk on the moon?\nAssistant:",
		llms.WithTemperature(0.8),
		llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
			fmt.Print(string(chunk))
			return nil
		}),
	)
	if err != nil {
		log.Fatal(err)
	}

	_ = completion
}

```

// # openai_chat_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	llm, err := openai.NewChat()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, []schema.ChatMessage{
		schema.SystemChatMessage{Content: "Hello, I am a friendly chatbot. I love to talk about movies, books and music. Answer in long form yaml."},
		schema.HumanChatMessage{Content: "What would be a good company name a company that makes colorful socks?"},
	}, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Print(string(chunk))
		return nil
	}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}

```

// # openai_completion_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
)

func main() {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, "The first man to walk on the moon",
		llms.WithTemperature(0.8),
		llms.WithStopWords([]string{"Armstrong"}),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}

```

// # openai_function_call_example.go Contents:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	llm, err := openai.NewChat(openai.WithModel("gpt-3.5-turbo-0613"))
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, []schema.ChatMessage{
		schema.HumanChatMessage{Content: "What is the weather like in Boston?"},
	}, llms.WithFunctions(functions))
	if err != nil {
		log.Fatal(err)
	}

	if completion.FunctionCall != nil {
		fmt.Printf("Function call: %v\n", completion.FunctionCall)
	}
}

func getCurrentWeather(location string, unit string) (string, error) {
	weatherInfo := map[string]interface{}{
		"location":    location,
		"temperature": "72",
		"unit":        unit,
		"forecast":    []string{"sunny", "windy"},
	}
	b, err := json.Marshal(weatherInfo)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

var functions = []llms.FunctionDefinition{
	{
		Name:        "getCurrentWeather",
		Description: "Get the current weather in a given location",
		Parameters:  json.RawMessage(`{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}`),
	},
}

```

// # openai_function_call_example.go Contents:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/jsonschema"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	llm, err := openai.NewChat(openai.WithModel("gpt-3.5-turbo-0613"))
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, []schema.ChatMessage{
		schema.HumanChatMessage{Content: "What is the weather going to be like in Boston?"},
	}, llms.WithFunctions(functions), llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Printf("Received chunk: %s\n", chunk)
		return nil
	}))
	if err != nil {
		log.Fatal(err)
	}

	if completion.FunctionCall != nil {
		fmt.Printf("Function call: %+v\n", completion.FunctionCall)
	}
	fmt.Println(completion.Content)
}

func getCurrentWeather(location string, unit string) (string, error) {
	weatherInfo := map[string]interface{}{
		"location":    location,
		"temperature": "72",
		"unit":        unit,
		"forecast":    []string{"sunny", "windy"},
	}
	b, err := json.Marshal(weatherInfo)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// json.RawMessage(`{"type": "object", "properties": {"location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["location"]}`),

var functions = []llms.FunctionDefinition{
	{
		Name:        "getCurrentWeather",
		Description: "Get the current weather in a given location",
		Parameters: jsonschema.Definition{
			Type: jsonschema.Object,
			Properties: map[string]jsonschema.Definition{
				"rationale": {
					Type:        jsonschema.String,
					Description: "The rationale for choosing this function call with these parameters",
				},
				"location": {
					Type:        jsonschema.String,
					Description: "The city and state, e.g. San Francisco, CA",
				},
				"unit": {
					Type: jsonschema.String,
					Enum: []string{"celsius", "fahrenheit"},
				},
			},
			Required: []string{"rationale", "location"},
		},
	},
	{
		Name:        "getTomorrowWeather",
		Description: "Get the predicted weather in a given location",
		Parameters: jsonschema.Definition{
			Type: jsonschema.Object,
			Properties: map[string]jsonschema.Definition{
				"rationale": {
					Type:        jsonschema.String,
					Description: "The rationale for choosing this function call with these parameters",
				},
				"location": {
					Type:        jsonschema.String,
					Description: "The city and state, e.g. San Francisco, CA",
				},
				"unit": {
					Type: jsonschema.String,
					Enum: []string{"celsius", "fahrenheit"},
				},
			},
			Required: []string{"rationale", "location"},
		},
	},
	{
		Name:        "getSuggestedPrompts",
		Description: "Given the user's input prompt suggest some related prompts",
		Parameters: jsonschema.Definition{
			Type: jsonschema.Object,
			Properties: map[string]jsonschema.Definition{
				"rationale": {
					Type:        jsonschema.String,
					Description: "The rationale for choosing this function call with these parameters",
				},
				"suggestions": {
					Type: jsonschema.Array,
					Items: &jsonschema.Definition{
						Type:        jsonschema.String,
						Description: "A suggested prompt",
					},
				},
			},
			Required: []string{"rationale", "suggestions"},
		},
	},
}

```

// # openai_gpt4_turbo.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	llm, err := openai.NewChat(openai.WithModel("gpt-4-1106-preview"))
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, []schema.ChatMessage{
		schema.SystemChatMessage{Content: "You are a company branding design wizard."},
		schema.HumanChatMessage{Content: "What would be a good company name a company that makes colorful socks?"},
	}, llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Print(string(chunk))
		return nil
	}),
	)
	if err != nil {
		log.Fatal(err)
	}
	_ = completion
}

```

// # pinecone_vectorstore_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/tmc/langchaingo/embeddings/openai"
	"github.com/tmc/langchaingo/schema"
	"github.com/tmc/langchaingo/vectorstores"
	"github.com/tmc/langchaingo/vectorstores/pinecone"
)

func main() {
	// Create an embeddings client using the OpenAI API. Requires environment variable OPENAI_API_KEY to be set.
	e, err := openai.NewOpenAI()
	if err != nil {
		log.Fatal(err)
	}

	ctx := context.Background()

	// Create a new Pinecone vector store.
	store, err := pinecone.New(
		ctx,
		pinecone.WithProjectName("YOUR_PROJECT_NAME"),
		pinecone.WithIndexName("YOUR_INDEX_NAME"),
		pinecone.WithEnvironment("YOUR_ENVIRONMENT"),
		pinecone.WithEmbedder(e),
		pinecone.WithAPIKey("YOUR_API_KEY"),
		pinecone.WithNameSpace(uuid.New().String()),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Add documents to the Pinecone vector store.
	err = store.AddDocuments(context.Background(), []schema.Document{
		{
			PageContent: "Tokyo",
			Metadata: map[string]any{
				"population": 38,
				"area":       2190,
			},
		},
		{
			PageContent: "Paris",
			Metadata: map[string]any{
				"population": 11,
				"area":       105,
			},
		},
		{
			PageContent: "London",
			Metadata: map[string]any{
				"population": 9.5,
				"area":       1572,
			},
		},
		{
			PageContent: "Santiago",
			Metadata: map[string]any{
				"population": 6.9,
				"area":       641,
			},
		},
		{
			PageContent: "Buenos Aires",
			Metadata: map[string]any{
				"population": 15.5,
				"area":       203,
			},
		},
		{
			PageContent: "Rio de Janeiro",
			Metadata: map[string]any{
				"population": 13.7,
				"area":       1200,
			},
		},
		{
			PageContent: "Sao Paulo",
			Metadata: map[string]any{
				"population": 22.6,
				"area":       1523,
			},
		},
	})
	if err != nil {
		log.Fatal(err)
	}

	// Search for similar documents.
	docs, err := store.SimilaritySearch(ctx, "japan", 1)
	fmt.Println(docs)

	// Search for similar documents using score threshold.
	docs, err = store.SimilaritySearch(ctx, "only cities in south america", 10, vectorstores.WithScoreThreshold(0.80))
	fmt.Println(docs)

	// Search for similar documents using score threshold and metadata filter.
	filter := map[string]interface{}{
		"$and": []map[string]interface{}{
			{
				"area": map[string]interface{}{
					"$gte": 1000,
				},
			},
			{
				"population": map[string]interface{}{
					"$gte": 15.5,
				},
			},
		},
	}

	docs, err = store.SimilaritySearch(ctx, "only cities in south america",
		10,
		vectorstores.WithScoreThreshold(0.80),
		vectorstores.WithFilters(filter))
	fmt.Println(docs)
}

```

// # postgresql_database_chain.go Contents:

```go
package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools/sqldatabase"
	_ "github.com/tmc/langchaingo/tools/sqldatabase/postgresql"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func makeSample(dsn string) {
	db, err := sql.Open("pgx", dsn)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	sqlStmt := `
	CREATE TABLE IF NOT EXISTS foo (id integer not null primary key, name text);
	delete from foo;
	CREATE TABLE IF NOT EXISTS foo1 (id integer not null primary key, name text);
	delete from foo1;
	`
	_, err = db.Exec(sqlStmt)
	if err != nil {
		log.Fatal(err)
	}

	tx, err := db.Begin()
	if err != nil {
		log.Fatal(err)
	}
	stmt, err := tx.Prepare("insert into foo(id, name) values($1, $2)")
	if err != nil {
		log.Fatal(err)
	}
	defer stmt.Close()
	for i := 0; i < 100; i++ {
		_, err = stmt.Exec(i, fmt.Sprintf("Foo %03d", i))
		if err != nil {
			log.Fatal(err)
		}
	}

	stmt1, err := tx.Prepare("insert into foo1(id, name) values($1, $2)")
	if err != nil {
		log.Fatal(err)
	}
	defer stmt1.Close()
	for i := 0; i < 200; i++ {
		_, err = stmt1.Exec(i, fmt.Sprintf("Foo1 %03d", i))
		if err != nil {
			log.Fatal(err)
		}
	}

	err = tx.Commit()
	if err != nil {
		log.Fatal(err)
	}
}

func run() error {
	llm, err := openai.New()
	if err != nil {
		return err
	}

	dsn := os.Getenv("LANGCHAINGO_POSTGRESQL")

	makeSample(dsn)

	db, err := sqldatabase.NewSQLDatabaseWithDSN("pgx", dsn, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	sqlDatabaseChain := chains.NewSQLDatabaseChain(llm, 100, db)
	ctx := context.Background()
	out, err := chains.Run(ctx, sqlDatabaseChain, "Return all rows from the foo table where the ID is less than 23.")
	if err != nil {
		return err
	}
	fmt.Println(out)

	input := map[string]any{
		"query":              "Return all rows that the ID is less than 23.",
		"table_names_to_use": []string{"foo"},
	}
	out, err = chains.Predict(ctx, sqlDatabaseChain, input)
	if err != nil {
		return err
	}
	fmt.Println(out)

	out, err = chains.Run(ctx, sqlDatabaseChain, "Which table has more data, foo or foo1$")
	if err != nil {
		return err
	}
	fmt.Println(out)
	return err
}

```

// # propmts_with_partial_example.go Contents:

```go
package main

import (
	"fmt"
	"log"

	"github.com/tmc/langchaingo/prompts"
)

func main() {
	prompt := prompts.PromptTemplate{
		Template:       "{{.foo}}{{.bar}}",
		InputVariables: []string{"bar"},
		PartialVariables: map[string]any{
			"foo": "foo",
		},
		TemplateFormat: prompts.TemplateFormatGoTemplate,
	}
	result, err := prompt.Format(map[string]any{
		"bar": "baz",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}

```

// # propmts_with_partial_func_example.go Contents:

```go
package main

import (
	"fmt"
	"log"
	"time"

	"github.com/tmc/langchaingo/prompts"
)

func main() {
	prompt := prompts.PromptTemplate{
		Template:       "Tell me a {{.adjective}} joke about the day {{.date}}",
		InputVariables: []string{"adjective"},
		PartialVariables: map[string]any{
			"date": func() string {
				return time.Now().Format("January 02, 2006")
			},
		},
		TemplateFormat: prompts.TemplateFormatGoTemplate,
	}
	result, err := prompt.Format(map[string]any{
		"adjective": "funny",
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
}

```

// # sequential_chain_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

func main() {
	simpleSequentialChainExample()
	sequentialChainExample()
}

func simpleSequentialChainExample() {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}

	template1 := `
        You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
        Title: {{.title}}
        Playwright: This is a synopsis for the above play:
    `
	chain1 := chains.NewLLMChain(llm, prompts.NewPromptTemplate(template1, []string{"title"}))

	template2 := `
        You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
        Play Synopsis:
        {{.synopsis}}
        Review from a New York Times play critic of the above play:
    `
	chain2 := chains.NewLLMChain(llm, prompts.NewPromptTemplate(template2, []string{"synopsis"}))

	simpleSeqChain, err := chains.NewSimpleSequentialChain([]chains.Chain{chain1, chain2})
	if err != nil {
		log.Fatal(err)
	}

	res, err := chains.Run(context.Background(), simpleSeqChain, "Tragedy at sunset on the beach")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res)
}

func sequentialChainExample() {
	llm, err := openai.New()
	if err != nil {
		log.Fatal(err)
	}

	template1 := `
	You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.
	Title: {{.title}}
	Era: {{.era}}
	Playwright: This is a synopsis for the above play:
	`
	chain1 := chains.NewLLMChain(llm, prompts.NewPromptTemplate(template1, []string{"title", "era"}))
	chain1.OutputKey = "synopsis"

	template2 := `
		You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
		Play Synopsis:
		{{.synopsis}}
		Review from a New York Times play critic of the above play:
	`
	chain2 := chains.NewLLMChain(llm, prompts.NewPromptTemplate(template2, []string{"synopsis"}))
	chain2.OutputKey = "review"

	sequentialChain, err := chains.NewSequentialChain([]chains.Chain{chain1, chain2}, []string{"title", "era"}, []string{"review"})
	if err != nil {
		log.Fatal(err)
	}

	res, err := chains.Call(context.Background(), sequentialChain, map[string]any{"title": "Tragedy at sunset on the beach", "era": "Victorian"})
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(res["review"])
}

```

// # sql_database_chain.go Contents:

```go
package main

import (
	"context"
	"database/sql"
	"fmt"
	"log"
	"os"

	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools/sqldatabase"
	_ "github.com/tmc/langchaingo/tools/sqldatabase/sqlite3"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func makeSample(dsn string) {
	db, err := sql.Open("sqlite3", dsn)
	if err != nil {
		log.Fatal(err)
	}
	defer db.Close()

	sqlStmt := `
	create table foo (id integer not null primary key, name text);
	delete from foo;
	create table foo1 (id integer not null primary key, name text);
	delete from foo1;
	`
	_, err = db.Exec(sqlStmt)
	if err != nil {
		log.Fatal(err)
	}

	tx, err := db.Begin()
	if err != nil {
		log.Fatal(err)
	}
	stmt, err := tx.Prepare("insert into foo(id, name) values(?, ?)")
	if err != nil {
		log.Fatal(err)
	}
	defer stmt.Close()
	for i := 0; i < 100; i++ {
		_, err = stmt.Exec(i, fmt.Sprintf("Foo %03d", i))
		if err != nil {
			log.Fatal(err)
		}
	}

	stmt1, err := tx.Prepare("insert into foo1(id, name) values(?, ?)")
	if err != nil {
		log.Fatal(err)
	}
	defer stmt1.Close()
	for i := 0; i < 200; i++ {
		_, err = stmt1.Exec(i, fmt.Sprintf("Foo1 %03d", i))
		if err != nil {
			log.Fatal(err)
		}
	}

	err = tx.Commit()
	if err != nil {
		log.Fatal(err)
	}
}

func run() error {
	llm, err := openai.New()
	if err != nil {
		return err
	}

	const dsn = "./foo.db"
	os.Remove(dsn)
	defer os.Remove(dsn)

	makeSample(dsn)

	db, err := sqldatabase.NewSQLDatabaseWithDSN("sqlite3", dsn, nil)
	if err != nil {
		return err
	}
	defer db.Close()

	sqlDatabaseChain := chains.NewSQLDatabaseChain(llm, 100, db)
	ctx := context.Background()
	out, err := chains.Run(ctx, sqlDatabaseChain, "Return all rows from the foo table where the ID is less than 23.")
	if err != nil {
		return err
	}
	fmt.Println(out)

	input := map[string]any{
		"query":              "Return all rows that the ID is less than 23.",
		"table_names_to_use": []string{"foo"},
	}
	out, err = chains.Predict(ctx, sqlDatabaseChain, input)
	if err != nil {
		return err
	}
	fmt.Println(out)

	out, err = chains.Run(ctx, sqlDatabaseChain, "Which table has more data, foo or foo1?")
	if err != nil {
		return err
	}
	fmt.Println(out)
	return err
}

```

// # vertexai_palm_chat_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms/vertexai"
	"github.com/tmc/langchaingo/schema"
)

func main() {
	llm, err := vertexai.NewChat()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	question := schema.HumanChatMessage{
		Content: "What would be a good company name a company that makes colorful socks?",
	}
	fmt.Println("---ASK---")
	fmt.Println(question.GetContent())
	messages := []schema.ChatMessage{question}
	completion, err := llm.Call(ctx, messages)
	if err != nil {
		log.Fatal(err)
	}

	response := completion
	fmt.Println("---RESPONSE---")
	fmt.Println(response)

	// keep track of conversation
	messages = append(messages, response)

	question = schema.HumanChatMessage{
		Content: "Any other recommendation on how to get started with the company ?",
	}
	fmt.Println("---ASK---")
	fmt.Println(question.GetContent())
	messages = append(messages, question)

	completion, err = llm.Call(ctx, messages)
	if err != nil {
		log.Fatal(err)
	}

	response = completion
	fmt.Println("---RESPONSE---")
	fmt.Println(response)
}

```

// # vertexai_palm_completion_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms/vertexai"
)

func main() {
	llm, err := vertexai.New()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	completion, err := llm.Call(ctx, "The first man to walk on the moon")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(completion)
}

```

// # vertexai_palm_embeddings_example.go Contents:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/llms/vertexai"
)

func main() {
	llm, err := vertexai.NewChat()
	if err != nil {
		log.Fatal(err)
	}
	ctx := context.Background()
	embeddings, err := llm.CreateEmbedding(ctx, []string{"I am a human"})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(embeddings)
}

```

// # main.go Contents:

```go
package main

import (
	"context"
	"fmt"

	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/chains"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/tools"
	"github.com/tmc/langchaingo/tools/zapier"
)

func main() {
	ctx := context.Background()

	llm, err := openai.New()
	if err != nil {
		panic(err)
	}

	// set env variable ZAPIER_NLA_API_KEY to your Zapier API key

	// get all the available zapier NLA Tools
	tks, err := zapier.Toolkit(ctx, zapier.ToolkitOpts{
		// APIKey: "SOME_KEY_HERE", Or pass in a key here
		// AccessToken: "ACCESS_TOKEN", this is if your using OAuth
	})
	if err != nil {
		panic(err)
	}

	agentTools := []tools.Tool{
		// define tools here
	}
	// add the zapier tools to the existing agentTools
	agentTools = append(agentTools, tks...)

	// Initialize the agent
	executor, err := agents.Initialize(
		llm,
		agentTools,
		agents.ZeroShotReactDescription,
		agents.WithMaxIterations(3),
	)
	if err != nil {
		panic(err)
	}

	// run a chain with the executor and defined input
	input := "Get the last email from noreply@github.com"
	answer, err := chains.Run(context.Background(), executor, input)
	if err != nil {
		panic(err)
	}
	fmt.Println(answer)
}

```

