import click
import jax


def bad():
    """
    The same key will generate the exact same sequence every time it is used, on any device. I ran this
    program multiple times on my own computer as well as in a Google Colab notebook and I got the exact
    same sequence -

    [0.28290725 0.57700586 0.9130496  0.91475654 0.572966   0.92008424
    0.6318501  0.20412707 0.94751775 0.41770518]
    """
    key = jax.random.key(786)
    nums = jax.random.uniform(key, (10,))
    print(nums)

    nums = jax.random.uniform(key, (10,))
    print(nums)


def good():
    key = jax.random.key(786)
    print(f"key: {key}")

    for _ in range(2):
        # I can generate multiple new keys and use them on different devices/nodes
        # keys = jax.random.split(key, 10)
        newkey, subkey = jax.random.split(key)
        print(f"newkey: {newkey}")
        print(f"subkey: {subkey}")

        # key has now been used, so we throw it away
        key = None

        # Generate the random numbers using subkey
        nums = jax.random.uniform(subkey, (10,))
        print(nums)

        # In the next round, generate the subkey with newkey
        key = newkey


@click.command()
@click.option("--good", "func", flag_value="good")
@click.option("--bad", "func", flag_value="bad")
def main(func):
    if func == "good":
        good()
    elif func == "bad":
        bad()
    else:
        raise ValueError("No such function!")


if __name__ == "__main__":
    main()
