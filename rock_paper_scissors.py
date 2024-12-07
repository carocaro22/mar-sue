import random
from enum import Enum

class Pick(Enum):
    Rock = 1
    Paper = 2
    Scissor = 3

print("Please type \n r: Rock, p: Paper, s: Scissor")
total = 0
won = 0
lost = 0
games = 0
while(games < 20):
    a = input()
    b = random.randint(1,3)
    match a: 
        case 'r': 
            a = Pick.Rock;
            print("You picked Rock")
            match b: 
                case Pick.Paper.value: 
                    print("Computer picked Paper")
                    total += 1
                    lost += 1
                    games += 1
                    print("You lose")
                case Pick.Scissor.value: 
                    print("Computer picked Scissor")
                    total += 1
                    won += 1
                    games += 1
                    print("You won")
                case Pick.Rock.value: 
                    print("Computer picked Rock")
                    print("It's a tie")
        case 'p': 
            a = Pick.Paper
            print("You picked Paper")
            match b: 
                case Pick.Paper.value: 
                    print("Computer picked Paper")
                    print("It's a tie")
                case Pick.Scissor.value: 
                    print("Computer picked Scissor")
                    total += 1
                    lost += 1
                    games += 1
                    print("You lost")
                case Pick.Rock.value: 
                    print("Computer picked Rock")
                    total += 1
                    won += 1
                    games += 1
                    print("You won")
        case 's': 
            a = Pick.Scissor
            print("You picked Scissor")
            match b: 
                case Pick.Paper.value: 
                    print("Computer picked Paper")
                    total += 1
                    won += 1
                    games += 1
                    print("You won")
                case Pick.Scissor.value: 
                    print("Computer picked Scissor")
                    print("It's a tie")
                case Pick.Rock.value: 
                    print("Computer picked Rock")
                    total += 1
                    lost += 1
                    games += 1
                    print("You lost")
        case _: 
            print("Please pick a valid option")

# evaluation: 
print(f"Total games: {total}")
print(f"Won: {won}")
print(f"Lost: {lost}")
print(f"Win rate: {won/total*100}%")