import textgrad as tg
from hone import get_instill_engine


def main():

    engine = get_instill_engine(
        "gpt-3.5-turbo",
        namespace_id="george_strong",
        pipeline_id="textgrad-engine"
    )

    response = engine.generate("What is the capital of France?")
    print("Response:", response)

    x = tg.Variable(
        "A sntence with a typo",
        role_description="The input sentence",
        requires_grad=True
    )

    print("Initial sentence:", x.value)

    system_prompt = tg.Variable(
        "Evaluate the correctness of this sentence",
        role_description="The system prompt"
    )
    loss = tg.TextLoss(system_prompt, engine=engine)

    optimizer = tg.optimizer.TextualGradientDescent(
        parameters=[x],
        engine=engine
    )

    loss_value = loss(x)
    loss_value.backward(engine)
    optimizer.step()

    print("Loss:", loss_value.value)
    print("Gradient:", x.gradients)
    print("Optimized sentence:", x.value)

    optimizer.zero_grad()


if __name__ == "__main__":
    main()
