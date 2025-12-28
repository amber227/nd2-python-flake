import sys
from pygifsicle import optimize

def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} input.gif output.gif")
        sys.exit(1)
    input_gif = sys.argv[1]
    output_gif = sys.argv[2]
    optimize(input_gif, output_gif)
    print(f"Optimized GIF saved to {output_gif}")

if __name__ == "__main__":
    main()
