import json

participants = ["zheer", "yun", "wu", "baying", "chixiang", "quijia", "ruibo", "yiren", "zhaoyang", "hao"]
userinput_list = ["Tap", "Swipe", "Knock", "Slap"]
activity_list = userinput_list
thing_to_activity = {"table":userinput_list,
                    "drawer":userinput_list,
                    # "cuttingboard":["Chopping", "Slicing", "Tenderlizing", "Stirring", "Rolling"]
                     }
things = ["table", "drawer"]

all_data = []
for thing in things:
    for participant in participants:
        for a in thing_to_activity[thing]:
            file_name ='./activity_data/'+thing + "/" + participant + "/"+ a +'.json'
            data = None
            with open(file_name, 'r+') as file:
                print(file)
                data = json.load(file)
                print(len(data))
                if len(data) != 10:
                	print(thing)
                	print(participant)
                	print(a)
                	print(len(data))
                data = data[:10]
                
            if data:
                with open(file_name, "w+") as file:
                	print(file)
                	json.dump(data, file, allow_nan = True)
