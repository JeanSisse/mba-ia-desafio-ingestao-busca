from search import search_prompt

def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado.")
    print("Digite 'exit' ou 'quit' para sair.\n")

    while True:
        try:
            user_input = input("Pergunte FCAI: ").strip()

            if user_input.lower() in ("exit", "quit"):
                print("Até mais!")
                break

            if not user_input:
                continue

            answer = chain.invoke({"pergunta": user_input})

            print(f"\nAssistente: {answer}\n")

        except KeyboardInterrupt:
            print("\nChat encerrado.")
            break

        except Exception as error:
            print(f"\nErro ao processar a pergunta: {error}\n")

if __name__ == "__main__":
    main()