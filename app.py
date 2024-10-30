from workflow import workflow


app = workflow.compile()

if __name__ == "__main__":

    question = input("You: ")
    result = app.invoke({"question": question})
    print(f"Documents: \n{result['documents']} \n===================================")
    print(f"Question: {result['question']} \n===================================")
    print(f"Answer: {result['answer']} \n===================================")
    print(f"Chat history: {result['chat_history']} \n===================================")
    print(f"Query search: {result['query_search']}")