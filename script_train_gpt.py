from happytransformer import HappyGeneration, GENSettings, GENTrainArgs

args = GENTrainArgs(
    learning_rate=1e-5, 
    batch_size=32,
    num_train_epochs=3,
)

happy_gen = HappyGeneration("gpt-neo-1.3B", "EleutherAI/gpt-neo-1.3B")
happy_gen.train("texts_1000.txt", args=args)
happy_gen.save("./gpt-overton")

print("Training and save over !")