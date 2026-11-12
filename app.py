import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import re

# Configuração da Página
st.set_page_config(
    page_title="Analisador Orçamentário & RAG com Groq",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- FUNÇÕES DE TRATAMENTO DE DADOS ---

def limpar_numero(val):
    """Converte strings numéricas (padrão BR ou US) para float com segurança."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    if not s or s.lower() in ['nan', 'none', '-']:
        return 0.0
    # Se tiver vírgula e ponto (ex: 1.234,56), remove ponto e troca vírgula
    if ',' in s and '.' in s:
        s = s.replace('.', '').replace(',', '.')
    elif ',' in s:
        s = s.replace(',', '.')
    s = re.sub(r'[^\d.-]', '', s)
    try:
        return float(s)
    except ValueError:
        return 0.0

@st.cache_data
def processar_planilha_orcamento(file_bytes, filename):
    """
    Lê a planilha Excel (como a do Coren SP) ou CSV, identifica o cabeçalho real,
    separa etapas de insumos e estrutura os dados para gráficos e RAG.
    """
    if filename.endswith('.csv'):
        # Tenta detectar separador automaticamente
        sample = file_bytes.decode('utf-8', errors='ignore')[:2048]
        sep = ';' if sample.count(';') > sample.count(',') else ','
        df_raw = pd.read_csv(io.BytesIO(file_bytes), sep=sep, header=None)
    else:
        df_raw = pd.read_excel(io.BytesIO(file_bytes), header=None)

    # Verifica se é uma planilha orçamentária padrão (procura a linha com 'Item' e 'Descrição')
    header_idx = None
    for idx, row in df_raw.iterrows():
        row_str = [str(x).strip().lower() for x in row.values]
        if 'item' in row_str and ('descrição' in row_str or 'descricao' in row_str):
            header_idx = idx
            break

    # Caso seja o formato específico de orçamento de obras (ex: Coren SP)
    if header_idx is not None:
        # Pula a linha de subcabeçalho (M.O. / MAT. / Total) logo abaixo de 'Item'
        data_start = header_idx + 2
        df = df_raw.iloc[data_start:].copy().reset_index(drop=True)

        # Mapeamento das 14 colunas padrão da planilha orçamentária
        colunas_orcamento = [
            'Item', 'Código', 'Banco', 'Descrição', 'Unidade', 'Quantidade',
            'Valor_Unit', 'Unit_BDI_MO', 'Unit_BDI_MAT', 'Unit_BDI_Total',
            'Total_MO', 'Total_MAT', 'Total_Geral', 'Peso_Percentual'
        ]
        # Ajusta caso o arquivo tenha exatamente ou mais/menos colunas
        df = df.iloc[:, :len(colunas_orcamento)]
        df.columns = colunas_orcamento[:df.shape[1]]

        # Remove linhas de rodapé (Totais ->, Total sem BDI, assinatura, vazias)
        df = df[df['Item'].notna()].copy()
        df['Item'] = df['Item'].astype(str).str.strip()
        # Mantém apenas linhas onde Item começa com dígito (ex: 1, 1.1, 14.2.1)
        df = df[df['Item'].str.match(r'^\d+(\.\d+)*$')].copy()

        # Limpeza numérica
        cols_num = [
            'Quantidade', 'Valor_Unit', 'Unit_BDI_MO', 'Unit_BDI_MAT',
            'Unit_BDI_Total', 'Total_MO', 'Total_MAT', 'Total_Geral', 'Peso_Percentual'
        ]
        for col in cols_num:
            if col in df.columns:
                df[col] = df[col].apply(limpar_numero)

        # Identifica se a linha é uma Etapa/Grupo (sem código/unidade) ou um Insumo/Serviço
        df['Tipo_Linha'] = np.where(
            df['Código'].isna() | (df['Código'].astype(str).str.strip() == 'nan'),
            'Etapa / Subetapa',
            'Item / Serviço'
        )
        df['Etapa_Principal'] = df['Item'].apply(lambda x: x.split('.')[0])
        
        # Mapeia o nome da Etapa Principal para cada subitem
        etapas_map = df[df['Item'].str.match(r'^\d+$')].set_index('Item')['Descrição'].to_dict()
        df['Nome_Etapa_Principal'] = df['Etapa_Principal'].map(etapas_map).fillna('Outros')

        return df, "orcamento"

    # Fallback para CSV/Excel tabular genérico
    else:
        if filename.endswith('.csv'):
            sample = file_bytes.decode('utf-8', errors='ignore')[:2048]
            sep = ';' if sample.count(';') > sample.count(',') else ','
            df_generic = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
        else:
            df_generic = pd.read_excel(io.BytesIO(file_bytes))
        df_generic = df_generic.dropna(how='all').reset_index(drop=True)
        return df_generic, "generico"


# --- MOTOR RAG (INDEXAÇÃO E RECUPERAÇÃO DE CONTEXTO) ---

class SpreadsheetRAG:
    def __init__(self, df: pd.DataFrame, tipo_planilha: str):
        self.df = df
        self.tipo = tipo_planilha
        self.documents = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self._build_index()

    def _build_index(self):
        """Transforma cada linha da planilha em um documento textual rico para busca."""
        docs = []
        for idx, row in self.df.iterrows():
            if self.tipo == "orcamento":
                if row['Tipo_Linha'] == 'Etapa / Subetapa':
                    txt = (
                        f"[ETAPA {row['Item']}] {row['Descrição']} | "
                        f"Valor Total da Etapa: R$ {row['Total_Geral']:,.2f} | "
                        f"Peso na Obra: {row['Peso_Percentual']*100:.2f}%"
                    )
                else:
                    txt = (
                        f"[ITEM {row['Item']}] Etapa: {row['Nome_Etapa_Principal']} | "
                        f"Código: {row['Código']} ({row['Banco']}) | "
                        f"Descrição: {row['Descrição']} | "
                        f"Unidade: {row['Unidade']} | Quantidade: {row['Quantidade']} | "
                        f"Unitário c/ BDI: R$ {row['Unit_BDI_Total']:,.2f} | "
                        f"Total Mão de Obra: R$ {row['Total_MO']:,.2f} | "
                        f"Total Material: R$ {row['Total_MAT']:,.2f} | "
                        f"Total Geral: R$ {row['Total_Geral']:,.2f}"
                    )
            else:
                parts = [f"{col}: {row[col]}" for col in self.df.columns if pd.notna(row[col])]
                txt = f"Linha {idx + 1} -> " + " | ".join(parts)
            docs.append(txt)

        self.documents = docs
        if docs:
            self.tfidf_matrix = self.vectorizer.fit_transform(docs)

    def get_global_summary(self) -> str:
        """Gera um resumo estatístico global para evitar erros de agregação no LLM."""
        if self.tipo == "orcamento":
            df_itens = self.df[self.df['Tipo_Linha'] == 'Item / Serviço']
            df_etapas = self.df[self.df['Item'].str.match(r'^\d+$')]
            
            total_geral = df_itens['Total_Geral'].sum()
            total_mo = df_itens['Total_MO'].sum()
            total_mat = df_itens['Total_MAT'].sum()

            etapas_resumo = "\n".join([
                f"- Etapa {r['Item']} ({r['Descrição']}): R$ {r['Total_Geral']:,.2f} ({r['Peso_Percentual']*100:.2f}%)"
                for _, r in df_etapas.iterrows()
            ])

            top5_itens = df_itens.nlargest(5, 'Total_Geral')
            top5_resumo = "\n".join([
                f"- Item {r['Item']} ({r['Descrição'][:60]}...): R$ {r['Total_Geral']:,.2f}"
                for _, r in top5_itens.iterrows()
            ])

            return (
                f"RESUMO GLOBAL DO ORÇAMENTO:\n"
                f"- Total Geral (com BDI): R$ {total_geral:,.2f}\n"
                f"- Total Mão de Obra (M.O.): R$ {total_mo:,.2f}\n"
                f"- Total Material (MAT.): R$ {total_mat:,.2f}\n"
                f"- Quantidade de Etapas Principais: {len(df_etapas)}\n"
                f"- Quantidade de Itens/Serviços: {len(df_itens)}\n\n"
                f"TOTAIS POR ETAPA PRINCIPAL:\n{etapas_resumo}\n\n"
                f"TOP 5 ITENS MAIS CAROS DA PLANILHA:\n{top5_resumo}"
            )
        else:
            desc = self.df.describe(include='all').to_string()
            return f"RESUMO ESTATÍSTICO DO DATASET ({len(self.df)} linhas):\n{desc[:2500]}"

    def retrieve(self, query: str, top_k: int = 15) -> str:
        """Recupera os trechos mais relevantes + resumo global."""
        if not self.documents:
            return "Nenhum dado disponível."

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Pega os top_k índices mais similares
        top_indices = similarities.argsort()[-top_k:][::-1]
        retrieved_docs = [self.documents[i] for i in top_indices if similarities[i] > 0]

        # Se a busca lexical não achar termos exatos, retorna amostra inicial
        if not retrieved_docs:
            retrieved_docs = self.documents[:top_k]

        contexto = (
            f"{self.get_global_summary()}\n\n"
            f"REGISTROS ESPECÍFICOS RECUPERADOS PARA A PERGUNTA:\n" +
            "\n".join(retrieved_docs)
        )
        return contexto


# --- INTERFACE STREAMLIT ---

st.title("📊 Análise Inteligente de Planilhas & Chat RAG (Groq)")
st.markdown("Faça upload de arquivos **CSV** ou **Excel (.xlsx)** para gerar dashboards interativos e conversar com seus dados.")

# Barra Lateral (Configurações & API Key)
with st.sidebar:
    st.header("🔑 Configuração Groq")
    groq_api_key = st.text_input(
        "Chave de API do Groq:",
        type="password",
        placeholder="gsk_...",
        help="Obtenha sua chave gratuita em console.groq.com"
    )

    modelo_groq = st.selectbox(
        "Modelo LLM (Groq):",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ],
        index=0
    )

    top_k_docs = st.slider(
        "Linhas de contexto no RAG (Top-K):",
        min_value=5,
        max_value=40,
        value=15,
        help="Quantidade de linhas mais relevantes enviadas ao modelo a cada pergunta."
    )

    st.divider()
    st.header("📂 Upload de Dados")
    uploaded_file = st.file_uploader(
        "Envie sua planilha (.csv ou .xlsx)",
        type=["csv", "xlsx"]
    )

    if st.button("🗑️ Limpar Histórico do Chat"):
        st.session_state.messages = []
        st.rerun()

# Corpo Principal
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    df, tipo_planilha = processar_planilha_orcamento(file_bytes, uploaded_file.name)

    # Inicializa o motor RAG
    rag_engine = SpreadsheetRAG(df, tipo_planilha)

    # Abas principais
    tab_dashboard, tab_dados, tab_rag = st.tabs([
        "📈 Dashboard & Gráficos",
        "📋 Tabela Estruturada",
        "💬 Conversa via RAG (Groq)"
    ])

    # ==========================================
    # ABA 1: DASHBOARD & GRÁFICOS
    # ==========================================
    with tab_dashboard:
        if tipo_planilha == "orcamento":
            df_itens = df[df['Tipo_Linha'] == 'Item / Serviço'].copy()
            df_etapas = df[df['Item'].str.match(r'^\d+$')].copy()

            # KPIs Principais
            total_geral = df_itens['Total_Geral'].sum()
            total_mo = df_itens['Total_MO'].sum()
            total_mat = df_itens['Total_MAT'].sum()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Total Geral (c/ BDI)", f"R$ {total_geral:,.2f}")
            col2.metric("👷 Total Mão de Obra", f"R$ {total_mo:,.2f}", f"{(total_mo/total_geral*100):.1f}%")
            col3.metric("🧱 Total Materiais", f"R$ {total_mat:,.2f}", f"{(total_mat/total_geral*100):.1f}%")
            col4.metric("📦 Total de Serviços", f"{len(df_itens)} itens", f"{len(df_etapas)} etapas")

            st.divider()

            # Gráficos Linha 1
            c1, c2 = st.columns([3, 2])
            with c1:
                st.subheader("Custo Total por Etapa da Obra")
                df_etapas_sorted = df_etapas.sort_values('Total_Geral', ascending=True)
                df_etapas_sorted['Label'] = df_etapas_sorted['Item'] + " - " + df_etapas_sorted['Descrição'].str.slice(0, 35)
                fig_etapas = px.bar(
                    df_etapas_sorted,
                    x='Total_Geral',
                    y='Label',
                    orientation='h',
                    text_auto='.2s',
                    color='Total_Geral',
                    color_continuous_scale='Blues',
                    labels={'Total_Geral': 'Valor Total (R$)', 'Label': 'Etapa'}
                )
                fig_etapas.update_layout(showlegend=False, height=480)
                st.plotly_chart(fig_etapas, use_container_width=True)

            with c2:
                st.subheader("Proporção: Mão de Obra vs. Material")
                fig_pizza = px.pie(
                    names=['Mão de Obra (M.O.)', 'Materiais (MAT.)'],
                    values=[total_mo, total_mat],
                    hole=0.45,
                    color_discrete_sequence=['#1f77b4', '#ff7f0e']
                )
                fig_pizza.update_traces(textinfo='percent+label')
                fig_pizza.update_layout(height=480)
                st.plotly_chart(fig_pizza, use_container_width=True)

            # Gráficos Linha 2
            st.divider()
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("Top 10 Serviços/Insumos Mais Caros")
                top10 = df_itens.nlargest(10, 'Total_Geral').sort_values('Total_Geral', ascending=True)
                top10['Desc_Curta'] = top10['Item'] + " - " + top10['Descrição'].str.slice(0, 42) + "..."
                fig_top10 = px.bar(
                    top10,
                    x='Total_Geral',
                    y='Desc_Curta',
                    orientation='h',
                    color='Banco',
                    text_auto='.2s',
                    labels={'Total_Geral': 'Total (R$)', 'Desc_Curta': 'Serviço'}
                )
                fig_top10.update_layout(height=450)
                st.plotly_chart(fig_top10, use_container_width=True)

            with c4:
                st.subheader("Distribuição de Custos por Banco de Referência")
                df_banco = df_itens.groupby('Banco', as_index=False)['Total_Geral'].sum().sort_values('Total_Geral', ascending=False)
                fig_banco = px.bar(
                    df_banco,
                    x='Banco',
                    y='Total_Geral',
                    text_auto='.2s',
                    color='Banco',
                    labels={'Total_Geral': 'Total (R$)', 'Banco': 'Base de Preços'}
                )
                fig_banco.update_layout(showlegend=False, height=450)
                st.plotly_chart(fig_banco, use_container_width=True)

        else:
            # Dashboard Dinâmico para CSV/Excel Genérico
            st.subheader("Explorador Visual Automático")
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            if num_cols and cat_cols:
                col_x = st.selectbox("Eixo Categórico (Agrupamento):", cat_cols)
                col_y = st.selectbox("Métrica Numérica:", num_cols)
                df_agg = df.groupby(col_x, as_index=False)[col_y].sum().nlargest(15, col_y)
                fig = px.bar(df_agg, x=col_x, y=col_y, color=col_x, text_auto='.2s')
                st.plotly_chart(fig, use_container_width=True)
            elif num_cols:
                col_y = st.selectbox("Coluna Numérica:", num_cols)
                fig = px.histogram(df, x=col_y)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("A planilha não possui colunas numéricas para geração automática de gráficos.")

    # ==========================================
    # ABA 2: TABELA ESTRUTURADA & EXPORTAÇÃO CSV
    # ==========================================
    with tab_dados:
        st.subheader("Dados Tratados e Estruturados")
        
        if tipo_planilha == "orcamento":
            filtro_tipo = st.radio(
                "Exibir:",
                ["Todos os registros", "Apenas Itens/Serviços", "Apenas Etapas/Subetapas"],
                horizontal=True
            )
            df_view = df.copy()
            if filtro_tipo == "Apenas Itens/Serviços":
                df_view = df_view[df_view['Tipo_Linha'] == 'Item / Serviço']
            elif filtro_tipo == "Apenas Etapas/Subetapas":
                df_view = df_view[df_view['Tipo_Linha'] == 'Etapa / Subetapa']
        else:
            df_view = df

        st.dataframe(df_view, use_container_width=True, height=500)

        # Botão para baixar a versão limpa em CSV
        csv_limpo = df_view.to_csv(index=False, sep=';', decimal=',').encode('utf-8-sig')
        st.download_button(
            label="📥 Baixar Planilha Limpa em CSV",
            data=csv_limpo,
            file_name="dados_tratados.csv",
            mime="text/csv"
        )

    # ==========================================
    # ABA 3: CONVERSA VIA RAG COM GROQ
    # ==========================================
    with tab_rag:
        st.subheader("🤖 Converse com sua Planilha (RAG + Groq)")
        
        if not groq_api_key:
            st.warning("⚠️ Insira sua **Chave de API do Groq** na barra lateral esquerda para ativar o chat.")
        else:
            if "messages" not in st.session_state:
                st.session_state.messages = []

            # Sugestões rápidas de perguntas
            st.caption("Sugestões de perguntas:")
            cols_sug = st.columns(3)
            sugestoes = [
                "Qual é o valor total da obra e a divisão entre mão de obra e material?",
                "Quais são os itens relacionados a exaustores e dutos no 1º Subsolo?",
                "Quais são as 3 etapas mais caras desse orçamento?"
            ]
            for i, sug in enumerate(sugestoes):
                if cols_sug[i].button(sug, use_container_width=True):
                    st.session_state.prompt_sugerido = sug

            # Renderiza histórico
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # Entrada do usuário
            user_prompt = st.chat_input("Faça uma pergunta sobre os itens, valores, etapas ou bancos da planilha...")
            if "prompt_sugerido" in st.session_state:
                user_prompt = st.session_state.pop("prompt_sugerido")

            if user_prompt:
                st.session_state.messages.append({"role": "user", "content": user_prompt})
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                # Recupera contexto via RAG
                contexto_rag = rag_engine.retrieve(user_prompt, top_k=top_k_docs)

                system_prompt = (
                    "Você é um engenheiro de custos e analista de dados especialista. "
                    "Responda às perguntas do usuário usando ESTRITAMENTE os dados da planilha fornecidos no contexto abaixo. "
                    "Formate valores monetários em Reais (R$ X.XXX,XX). "
                    "Se a informação não estiver no contexto recuperado, diga claramente que não encontrou na planilha.\n\n"
                    f"=== CONTEXTO DA PLANILHA ===\n{contexto_rag}"
                )

                with st.chat_message("assistant"):
                    try:
                        client = Groq(api_key=groq_api_key)
                        
                        # Monta histórico recente para manter coerência conversacional
                        api_messages = [{"role": "system", "content": system_prompt}]
                        for m in st.session_state.messages[-6:]:
                            api_messages.append({"role": m["role"], "content": m["content"]})

                        stream = client.chat.completions.create(
                            model=modelo_groq,
                            messages=api_messages,
                            temperature=0.1,
                            max_tokens=1500,
                            stream=True
                        )

                        def stream_generator():
                            for chunk in stream:
                                if chunk.choices[0].delta.content:
                                    yield chunk.choices[0].delta.content

                        resposta_completa = st.write_stream(stream_generator())
                        st.session_state.messages.append({"role": "assistant", "content": resposta_completa})

                        with st.expander("🔍 Ver trechos recuperados pelo RAG"):
                            st.text(contexto_rag)

                    except Exception as e:
                        st.error(f"Erro ao comunicar com a API do Groq: {e}")

else:
    st.info("👈 Comece fazendo o upload da planilha **`Coren SP - Sede - Adequação dos Subsolos.xlsx`** (ou qualquer arquivo `.csv`) na barra lateral.")
