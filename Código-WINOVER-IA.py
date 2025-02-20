import os
import pyodbc
import streamlit as st
import base64
from langchain_openai import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory

# -----------------------------------
# CONFIGURAÇÕES E CONEXÃO
# -----------------------------------
os.environ["OPENAI_API_KEY"] = "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"

db_connection_string = (
    "Driver={SQL Server};"
    "Server=WOCC34;"
    "Database=DESENVOLVIMENTO;"
    "UID=matheus.fernandes;"
    "PWD=Wocc@2025;"
)
llm = ChatOpenAI(
    model_name="gpt-4o-mini",
    temperature=0.1,
    max_tokens=1000
)

conn = pyodbc.connect(db_connection_string)
cursor = conn.cursor()

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

logo_base64 = get_image_base64("T:/15.Digitais/15. PYTHON/IA - TESTES/LOGOS E PANTONES/winover-logo-BR-VS - BRANCO.PNG")
assistant_base64 = get_image_base64("T:/15.Digitais/15. PYTHON/IA - TESTES/LOGOS E PANTONES/Wino_Mão_Cintura_PNG - cortado.png")

# -----------------------------------
# FUNÇÕES DE SUPORTE AO BANCO
# -----------------------------------
def get_db_metadata() -> str:
    metadata_info = "Tabelas disponíveis:\n"
    cursor.execute("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_TYPE = 'BASE TABLE'")
    tables = cursor.fetchall()
    for schema, table in tables:
        metadata_info += f"\n• {schema}.{table}\nColunas:\n"
        query = f"""SELECT COLUMN_NAME, DATA_TYPE 
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_SCHEMA = '{schema}' 
                    AND TABLE_NAME = '{table}'"""
        cursor.execute(query)
        columns = cursor.fetchall()
        for col_name, col_type in columns:
            metadata_info += f"  - {col_name} ({col_type})\n"
    return metadata_info

if "db_metadata" not in st.session_state:
    st.session_state.db_metadata = get_db_metadata()

def generate_sql_query(nl_query: str) -> str:
    metadata = get_db_metadata()
    prompt = (
        f"Você é um especialista em SQL Server e responde sempre em português. Metadados:\n{metadata}\n\n"
        f"Crie uma consulta SQL VÁLIDA para responder:\n{nl_query}\n"
        "REGRAS ESTRITAS:\n"
        "- NUNCA use crases (`), aspas ou markdown\n"
        "- Use [ ] apenas para nomes reservados\n"
        "- Sempre especifique o schema (ex: dbo.tabela)\n"
        "- Formato: SOMENTE SQL BRUTO, sem comentários\n"
    )
    response = llm.invoke(prompt).content
    if any(p in nl_query.lower() for p in ["código", "mostre o código"]):
        return response.split(';')[0].strip().replace('```sql', '').replace('```', '')
    return response.split(';')[0].strip().replace('```sql', '').replace('```', '')

def execute_sql_query(nl_query: str) -> str:
    if any(p in nl_query.lower() for p in ["código", "mostre o código"]):
        sql_code = generate_sql_query(nl_query)
        return f"Segue o código SQL:\n{sql_code}"
    try:
        sql_query = generate_sql_query(nl_query)
        print(f"[DEBUG] Consulta gerada: {sql_query}")
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        if not rows:
            return "Nenhum resultado encontrado."
        if len(rows[0]) == 1:
            return str(rows[0][0])
        return "\n".join([str(row) for row in rows])
    except Exception as e:
        return f"ERRO: {str(e)}"

def general_chat_tool(nl_query: str) -> str:
    return llm.invoke(nl_query).content

# -----------------------------------
# INTEGRAÇÃO DA MEMÓRIA DE CONVERSA
# -----------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -----------------------------------
# CONFIGURAÇÃO DO AGENTE COM DUAS FERRAMENTAS E MEMÓRIA
# -----------------------------------
agent = initialize_agent(
    tools=[
        Tool(
            name="SQL Tool",
            func=execute_sql_query,
            description="Executa consultas SQL no banco de dados. Use para obter dados específicos ou retornar o código SQL gerado."
        ),
        Tool(
            name="General Chat",
            func=general_chat_tool,
            description="Responde a perguntas gerais e pode tratar de assuntos diversos."
        )
    ],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    handle_parsing_errors=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "Olá! Em que posso ajudar hoje?"}
    ]

# -----------------------------------
# INTERFACE DE CHAT NO STREAMLIT
# -----------------------------------
with st.sidebar:
    # Injetamos CSS para estilizar o botão e empurrá-lo para a direita
    st.markdown(
        """
        <style>
        /* Força a sidebar a ter layout em coluna e ocupar 100% da altura */
        [data-testid="stSidebar"] > div:first-child {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        /* Container que empurra o botão para baixo */
        .limpar-conversa {
            margin-top: auto; 
            padding-bottom: 550px; /* Ajuste conforme necessário */
            display: flex;
            justify-content: flex-end; /* Move o botão para a direita */
        }
        /* Ajuste do botão para evitar ocupar largura total */
        .limpar-conversa button {
            width: auto !important;
            min-width: 100px; /* Ajuste conforme o tamanho desejado */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Chats")

    st.markdown("<div class='limpar-conversa'>", unsafe_allow_html=True)
    if st.button("Limpar"):
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Olá! Em que posso ajudar hoje?"}
        ]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .title-container {{
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }}
    .title-text {{
        color: #484D50;
        font-size: 5px;
        margin-right: 5px;
    }}
    .title-logo {{
        width: 50px;
    }}
    </style>
    <div class="title-container">
        <h1 class="title-text">Chat Wino</h1>
        <img src="data:image/png;base64,{logo_base64}" class="title-logo">
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    [data-testid="stChatInput"] textarea {
        background-color: #484D50 !important;
        color: #ffffff !important;
    }
    [data-testid="stChatInput"] label {
        color: #484D50 !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: #ffffff !important;
        opacity: 0.7;
    }
    footer {
        background-color: #484D50 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<div id='mainContent'>", unsafe_allow_html=True)
chat_container = st.empty()

def display_chat_history():
    with chat_container.container():
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f"""<div style='text-align: right; background-color: #484D50; color: white;
                    padding: 10px; border-radius: 10px; margin: 5px; max-width: 60%; float: right;'>
                    <strong>Você:</strong> {msg['content']}</div>
                    <div style='clear: both;'></div>""",
                    unsafe_allow_html=True
                )
            elif msg["role"] == "assistant":
                st.markdown(
                    f"""<div style='display: flex; align-items: flex-start; margin: 5px; clear: both;'>
                    <img src='data:image/png;base64,{assistant_base64}' 
                    style='width:50px; height:50px; border-radius:50%; margin-right:10px;'>
                    <div style='background-color: #484D50; color: white; padding: 10px; border-radius: 20px; max-width: 60%;'>
                    <strong>Assistente Wino:</strong> {msg['content']}</div></div>""",
                    unsafe_allow_html=True
                )

display_chat_history()



user_input = st.chat_input("Digite sua mensagem...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    response = agent({"input": user_input})["output"]
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    display_chat_history()

st.markdown("</div>", unsafe_allow_html=True)
