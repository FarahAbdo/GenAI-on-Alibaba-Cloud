# GenAI on Alibaba-Cloud

This repository contains the source code for the **"GenAI on Alibaba Cloud"** blog series.

It demonstrates how to build powerful Generative AI applications using **Alibaba Cloud Model Studio** and the **Qwen (Tongyi Qianwen)** Large Language Model family. The code focuses on using the **OpenAI-Compatible API**, making it easy for developers to integrate Qwen into existing workflows.

##  Series Overview

| Episode | Topic | File | Description |
| :--- | :--- | :--- | :--- |
| **01** | **Hello Qwen** | `hello_qwen.py` | Basic setup, API connection, and chatting with Qwen-Plus. |
| **02** | **RAG (Chat with PDF)** | `pdf_rag.py` | Building a "Chat with Data" system using Embeddings and Vector Search. |
| **03** | **AI Agents** | `agent_qwen.py` | Teaching Qwen to use external tools via Function Calling. |

##  Prerequisites

1.  **Alibaba Cloud Account:** [Sign up here](https://www.alibabacloud.com/).
2.  **Model Studio API Key:** Get your key from the [Alibaba Cloud Console](https://modelstudio.console.alibabacloud.com/).
3.  **Python 3.8+** installed on your machine.

##  Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/FarahAbdo/GenAI-on-Alibaba-Cloud.git
    cd GenAI-on-Alibaba-Cloud
    ```

2.  **Install dependencies:**
    ```bash
    pip install openai python-dotenv numpy pdfplumber
    ```

3.  **Configure your API Key:**
    Create a file named `.env` in the root folder and add your key:
    ```env
    DASHSCOPE_API_KEY=sk-your_actual_key_here
    ```

> **Note on Regions:** This code is configured for the **International (Singapore)** endpoint to ensure global access. If you are using the China endpoint, update the `base_url` in the Python scripts.

---

##  Resources

*   **Alibaba Cloud Model Studio:** [Official Documentation](https://www.alibabacloud.com/help/en/model-studio)
*   **Qwen Model Family:** [HuggingFace / ModelScope](https://huggingface.co/Qwen)
*   **OpenAI Python Library:** [PyPI](https://pypi.org/project/openai/)

## 📄 License
This project is open-source and available under the MIT License. Feel free to use this code in your own projects!
