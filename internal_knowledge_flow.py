
from metaflow import FlowSpec, step, Parameter, IncludeFile
import os, json, pandas as pd
from pathlib import Path
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
import mlflow
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from fuzzywuzzy import fuzz
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from common import rerank_documents
from langchain.text_splitter import RecursiveCharacterTextSplitter



class InternalKnowledgeFlow(FlowSpec):
    evaluate = Parameter("evaluate", default=False, type=bool, help="Run evaluation using ground truth QA pairs.")


    user_query = Parameter("user_query", help="The user query for the assistant")


    mlflow_tracking_uri = Parameter('mlflow_tracking_uri', help = 'mlflow tracking server' ,
                                    default = os.getenv('MLFLOW_TRACKING_URI', '127.0.0.1:5003'))

    @step
    def start(self):
        print("Starting Internal Knowledge Assistant Pipeline")

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        run = mlflow.start_run(run_name = current.run_id)
        slef.mlflow_run_id = run.info.run_id
        self.next(self.chunk_text)

    # Now let's create a chunking mechanism where we chunk the data into some predefined characters lengths and do the overlapping among 
    # each chunks so that there are context alignment among each chunk.



    @step
    def chunk_text(self):

        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        mlflow.start_run(run_id = self.mlflow_run_id)
        mlflow.log_param('chunk_size' , self.chunk_size)
        mlflow.log_param('overlap_size' , self.overlap)
        try:
            self.chunk_size = 800
            self.overlap = 200
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
            self.chunks = []
            chunk_val = 1

            for file_path in Path("Data").glob("*.*"):
                extension = file_path.suffix.lower().replace(".", "")
                filename = file_path.name
                try:
                    if extension == "txt":
                        with open(file_path, "r", encoding="utf-8") as f:
                            text = f.read()
                    elif extension == "csv":
                        text = pd.read_csv(file_path).to_string()
                    elif extension == "json":
                        with open(file_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            text = json.dumps(data, indent=2)

                    else:
                        continue

                    # Semantic chunking
                    chunks = splitter.split_text(text)
                    for chunk in chunks:
                        metadata = {
                            "filename": filename,
                            "filetype": extension,
                            "chunk_index": chunk_val
                        }
                        doc = Document(page_content=chunk, metadata=metadata)
                        self.chunks.append(doc)
                        chunk_val += 1
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
                    continue

            if not self.chunks:
                raise ValueError("No chunks created from data")
        except Exception as e:
            print(f"Chunking failed: {e}")
            raise ValueError("No Data Present for Chunking")

        self.next(self.embed_chunks)

    """In above step we have uploaded all the data first irrespective of their filetype. Then we have chunked the data and added chunked
    text alongwith it's metadata using Langchain compatible Document method. This will store the chunks in a form of list with text 
    as data as well as it's metadata as the index. Once done we then create the FAISS vector store using the text associated with chunk.
    once done langchain automatically attach the metadata associated with each embedded text, which if needed can be retreived using 
    metadata too. Note that Documnet method above is creating a each entity into 2 parts (page_content and metadata)"""

    @step
    def embed_chunks(self):
        self.embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # embed_documents will pull text from doc.page_content
        self.retriever = FAISS.from_documents(
            documents=self.chunks,
            embedding=self.embedding_function
        )
        # Save the retriever to disk so API can load it
        self.retriever.save_local("faiss_index")

        self.next(self.query_model)


    @step
    def query_model(self):
        from langchain.prompts import ChatPromptTemplate
        from langchain.chains import create_retrieval_chain
        from langchain.chains.combine_documents import create_stuff_documents_chain
        import csv

        llm = Ollama(model='mistral')

        prompt = ChatPromptTemplate.from_template(
            """
            [INST] You are a concise assistant. Answer the question using only the provided context. 
            If the answer is not in the context, respond with "I don't know".

            Context:
            {context}

            Question:
            {input}

            Answer: [/INST]
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = self.retriever.as_retriever(search_kwargs={'k': 5})
        qa_chain = create_retrieval_chain(retriever, document_chain)

        self.prompt = prompt

        if self.evaluate:
            score = 0
            total = 0
            results = []

            with open("Data/qa_eval.csv", newline ='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, 1):
                    query = row["question"]
                    expected = row["expected_answer"].lower().strip()
                    try:
                        result = qa_chain.invoke({'input': query})
                        response = result["answer"].lower().strip()
                        similarity_score = fuzz.partial_ratio(expected, response)
                        match = similarity_score >= 85

                        mlflow.log_text(response, f"response_{idx}.txt")
                        mlflow.log_text(expected, f"expected_{idx}.txt")
                        mlflow.log_metric(f"similarity_score_{idx}", similarity_score)

                        score += int(match)
                        total += 1
                        results.append((query, expected, response, match))
                    except Exception as e:
                        print(f"Query failed: {e}")
                        continue

            self.evaluation_accuracy = score / total if total else 0
            mlflow.log_metric("evaluation_accuracy", self.evaluation_accuracy)
            print(f"Evaluation Accuracy: {self.evaluation_accuracy:.2%}")
            self.result = f"Evaluation Accuracy: {self.evaluation_accuracy:.2%} on {total} queries"
            self.docs_text = "\n".join([r[2] for r in results])  # Combine all responses
            self.source_docs = []
            self.prompt_input = "Evaluation mode"
        else:
            try:
                # Get top 10 from FAISS
                retriever = self.retriever.as_retriever(search_kwargs={'k': 10})
                docs = retriever.invoke(self.user_query)

                # Rerank down to top 5 using query-aware scoring
                top_docs = rerank_documents(self.user_query, docs, top_n=5)

                # Run LLM on reranked documents manually
                response = document_chain.invoke({
                    "context": top_docs,
                    "input": self.user_query
                })

                # Store results
                self.result = response
                self.source_docs = top_docs
                self.prompt_input = self.user_query
                self.docs_text = '\n'.join([doc.page_content for doc in top_docs])

            except Exception as e:
                print(f"Query failed: {e}")
                self.result = "Error processing query"
                self.source_docs = []
                self.prompt_input = self.user_query
                self.docs_text = ""


        self.next(self.log_mlflow)


    @step
    def log_mlflow(self):
        mlflow.set_experiment("Internal Knowledge Assistant")

        with mlflow.start_run():
            mlflow.log_param("user_query", self.user_query)
            mlflow.log_param("prompt_template", self.prompt)
            mlflow.log_text(self.docs_text, "retrieved_documents.txt")
            mlflow.log_text(self.prompt_input, "final_prompt.txt")
            mlflow.log_text(self.result, "response.txt")

        print("Query Result:", self.result)
        self.next(self.end)


    @step
    def end(self):
        print("\nPipeline complete!")
        print("\nUser Query:", self.user_query)
        print("\nFinal Answer:\n", self.result)  # ✅ No indexing here!


if __name__ == "__main__":
    InternalKnowledgeFlow()
