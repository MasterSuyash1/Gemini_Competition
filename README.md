# DocuVid Genie

This is a Streamlit-based web application that allows users to interact with their PDF documents or YouTube videos by asking questions and receiving AI-generated responses. 
The application utilizes Google Generative AI embeddings and FAISS for vector embeddings to provide detailed and accurate answers based on the context of the documents or videos.

## Team

We are **Neural Ninjas**, a team of dedicated AI enthusiasts committed to creating innovative solutions. Our team members are:

- **[Suyash Srivastav](https://github.com/MasterSuyash1)**
- **[Swetha Suravajjula](https://github.com/swetha-suravajjula)**
- **[Yadvendra Garg](https://github.com/agi-yads)**

Together, we are working to bring you DocuVid Genie, an advanced app designed to enhance your interaction with PDFs and videos using state-of-the-art AI technology

## Features

- **Upload PDF Files**: Users can upload one or more PDF documents to be analyzed by the AI assistant.
- **YouTube Video Analysis**: Users can input a YouTube video link to extract the transcript and ask questions related to the video's content.
- **Contextual AI Responses**: The AI assistant provides detailed, context-aware answers based on the content of the PDF documents or video.
- **Suggested Questions**: The application can generate a list of suggested questions based on the content, helping users explore the document or video in depth.
- **Persistent Chat History**: The conversation history is maintained throughout the session, allowing users to refer back to previous interactions.

## Installation

To run this application locally, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/chat-with-pdfs-or-video.git
    cd chat-with-pdfs-or-video
    ```

2. **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables**:
    - Create a `.env` file in the root directory.
    - Add your Google API key:
      ```env
      GOOGLE_API_KEY=your_google_api_key
      ```

5. **Run the application**:
    ```bash
    streamlit run app.py
    ```

## Usage

### Chat with PDF Documents

1. Upload one or more PDF files using the file uploader in the sidebar.
2. Click on "Analyze" to process the documents.
3. Once analyzed, you can ask questions related to the content of the PDFs.
4. Optionally, enable "Suggested Questions" to get AI-generated questions based on the document content.

### Chat with YouTube Videos

1. Enter the YouTube video link in the sidebar.
2. The application will extract the transcript and analyze the content.
3. Ask questions related to the video's transcript.
4. Similar to PDFs, you can also enable "Suggested Questions" for videos.

### Suggested Questions

- After analyzing a document or video, you can enable the "Suggested Questions" option.
- Choose the number of questions you'd like to generate using the slider.
- The AI will generate questions based on the content, which you can select to ask directly.

## Dependencies

- **Streamlit**: For creating the web application interface.
- **PyPDF2**: To extract text from PDF files.
- **langchain**: For text splitting, embeddings, and chaining the prompts with AI responses.
- **FAISS**: For creating and storing vector embeddings.
- **YouTubeTranscriptAPI**: For extracting transcripts from YouTube videos.
- **Google Generative AI**: For generating AI responses and embeddings.

## Contributing

Feel free to open issues or submit pull requests if you'd like to contribute to this project.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- The application uses the [Google Generative AI API](https://cloud.google.com/generative-ai) for embeddings and generating AI responses.
- The YouTube transcript extraction is powered by [YouTubeTranscriptAPI](https://pypi.org/project/youtube-transcript-api/).
