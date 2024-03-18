import streamlit as st

# Título da página
st.title('Chat')

user_name = 'Usuário'

# Inicializando o histórico de chat
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Placeholder para mensagens
chat_history = st.empty()

# CSS para fixar a caixa de texto na parte inferior e estilo do histórico de mensagen
st.markdown("""
    <style>
        .chat-messages {
            height: 300px;
            overflow-y: auto;
            display: flex;
            flex-direction: column-reverse;
            border: 1px solid #ccc;
            padding: 10px;
            margin-bottom: 20px;
        }
        .fixed-bottom {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background-color: white;
        }
        /* Corrigir a posição do footer */
        .streamlit-footer {
            padding-bottom: 70px;
        }
    </style>
""", unsafe_allow_html=True)

with st.form("form_message", clear_on_submit=True):
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    # Caixa de texto para a mensagem dentro de um formulário
    user_message = st.text_input(f'{user_name}, digite sua mensagem:', key="message_input")

    # Botão para enviar mensagem
    send_message = st.form_submit_button('Enviar Mensagem')

    st.markdown('</div>', unsafe_allow_html=True)

if send_message and user_message:
        st.session_state.messages.append(f"{user_name}: {user_message}")

# Exibir mensagens (invertido, as novas mensagens aparecem no topo)
st.markdown('<div class="chat-messages">' + '<br>'.join(reversed(st.session_state.messages)) + '</div>', unsafe_allow_html=True)