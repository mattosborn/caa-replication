#!/usr/bin/env python3

import argparse
from caa.generate_steering_vectors import generate_steering_vectors
from caa.test_steering_vectors import test_steering_vectors
from caa.test_baseline import test_baseline

DEFAULT_MODEL = 'meta-llama/Llama-2-7b-chat-hf'

def main():
    parser = argparse.ArgumentParser(description='A CLI tool for steering vector management')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='HF model name')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    generate_parser = subparsers.add_parser('generate', help='Generate steering vectors')
    steering_parser = subparsers.add_parser('steering', help='Test steering vectors')
    baseline_parser = subparsers.add_parser('baseline', help='Test baseline')

    args = parser.parse_args()
    
    print(f'Using model: {args.model}')

    if args.command == 'generate':
        generate_steering_vectors(args.model)
    elif args.command == 'steering':
        test_steering_vectors(args.model)
    elif args.command == 'baseline':
        test_baseline(args.model)
    else:
        print('Invalid command. Please try again.')
        parser.print_help()

if __name__ == "__main__":
    main()
