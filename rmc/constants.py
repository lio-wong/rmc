START_PARSE_TOKEN, END_PARSE_TOKEN = "<START_LANGUAGE_TO_WEBPPL_CODE>", "<END_LANGUAGE_TO_WEBPPL_CODE>"
START_SINGLE_PARSE_TOKEN, END_SINGLE_PARSE_TOKEN  = "<BEGIN_CODE>", "<END_CODE>"
START_COMMENT = "// "
START_DEFINITION = "// DEFINE: "
START_CONDITION = "// CONDITION: "

WEBPPL_START_DEFINITION = "var"
WEBPPL_START_CONDITION = "condition()"
WEBPPL_START_QUERY = "query"

LIBRARY_FUNCTIONS_HEADER = """// LIBRARY_FUNCTIONS"""
LIBRARY_FUNCTIONS = """
var argMax = function(f, ar){
    return maxWith(f, ar)[0]
};

var mapMaxOverArray = function(f, ar){ 
    return maxWith(f, ar)[1]
}

var argMin = function(f, ar){
    return minWith(f, ar)[0]
};

var mapMinOverArray = function(f, ar){ 
    return minWith(f, ar)[1]
}

var mean = function(ar){ 
    return sum(ar) / ar.length
}
"""

TRANSLATIONS_HEADER = """
Translate each new sentence into a function of WebPPL code. Begin each translation with <BEGIN_CODE> and end each translation with <END_CODE>. Functions may call any library function, or any other function in the context delimited by the current <START_LANGUAGE_TO_WEBPPL_CODE> scope.
"""


sports_map = {
    "tug-of-war": {"description": """In this event, the athletes are competing in matches of tug-of-war. In each round, the team that wins the round depends on how hard the athletes collectively pull, based on their intrinsic strength modulated by other factors including how much effort they put in to that round.""",
                   "description_full": 
"""In this event, the athletes are competing in tug-of-war tournaments. Each tournament consists of a series of matches. In each match, athletes compete as part of a team. 

An athlete’s intrinsic strength remains constant throughout a tournament. An athlete neither gets stronger nor weaker between matches. You can assume that all matches take place on the same day.

Athletes also vary in the effort that they put into any given match. Most of the time, people pull with a moderately high amount of effort. Sometimes, an athlete won’t put in much effort and will pull with only a fraction of their strength. Other times, they may put in a lot of effort and pull extra hard, beyond what their intrinsic strength would suggest.

How hard a team pulls overall in any given match is determined by the total amount that all of the athletes on the team pull in that match. How hard each athlete pulls in a given match is determined by their intrinsic strength, modified by how much effort they put in (a lower fraction of their intrinsic strength if they don’t put in much effort, or even more than their strength if they put in more effort).

The team that pulls the hardest in a given match wins.
 """,
    "skill": "strength",
    "latent": "effort",
    "match": "match"},
    "canoe-race": {"description": """In this event, the athletes are competing in a series of canoe races. In each race, the team that wins depends on the average speed with which the athletes are able to row, based on their intrinsic strength modulated by other factors including how much effort they put in to that race.""",
                   "description_full": """
In this event, the athletes are competing in a series of canoe racing tournaments. Each tournament consists of a series of races. In each race, athletes compete as part of a team. 

An athlete's intrinsic strength remains constant throughout a tournament. An athlete neither gets stronger nor weaker between races. You can assume that all matches take place on the same day.

Athletes also vary in the effort that they put into any given race. Most of the time, people row with a moderately high amount of effort. Sometimes, an athlete won’t put in much effort and will row with only a fraction of their strength. Other times, they may put in a lot of effort and row extra hard, beyond what their intrinsic strength would suggest.

How fast a team rows overall in any given race is determined by the average rowing speed of each athlete. How fast an athlete rows in a given race is determined by their intrinsic strength, modified by how much effort they put in (a lower fraction of their intrinsic strength if they don’t put in much effort, or even more than their strength if they put in more effort).

The team that rows the fastest (highest team speed) in a given race wins.
                   """,
    "skill": "strength",
    "latent": "effort",
    "match": "race"},

    "diving": {"description": """In this event, teams of players are competing in rounds of a synchronized diving tournament. The overall dive difficulty attempted by any given team is determined by the least skilled athlete on the team. In each round of a synchronized dive, the team's overall score depends on their dive difficulty and how well matched the team members are in their execution in that particular round.""",
"description_full": """
In this event, the athletes are competing in a series of synchronized diving tournaments. Each tournament consists of a series of rounds. In each match, athletes compete as part of a team. 

In a given round, each team receives a score based on the difficulty of the dive and the execution of the dive. 

How difficult of a dive a team takes on varies in any given round. Teams can differ in the difficulty they attempt on any given round. The same team may try different levels of difficulty on different rounds. The dives each team takes in a given match are chosen before the tournament begins and are therefore independent of the outcome of the previous round.

A team’s execution score is based on the average skill of the athletes on the team and how difficult the attempted dive was. If the athletes are highly skilled, but the dive is very difficult, they may not execute well. Conversely, if athletes are highly skilled but the dive is not very difficult, the team may get a high execution score.

Note that an athlete’s intrinsic skill remains constant throughout a tournament. An athlete neither gets more nor less skilled between rounds. You can assume that all rounds take place on the same day.

A team’s overall score is determined by the sum of the difficulty and execution scores. Difficulty and execution scores are weighted equally. 

The team that gets the highest score in any given round wins.
""",
    "skill": "skill",
    "latent": "difficulty",
    "match": "round"},
    "biathalon": {"description": """In this event, teams of players are competing in rounds of a biathalon, a winter sport that combines cross-country skiing races and rifle shooting. In each round, the team that wins depends on the average speed with which the athletes are able to ski, based on their intrinsic strength, as well as each team member's shooting accuracy in that particular round.""",
"description_full": """
In this event, the athletes are competing in a series of biathlon relay tournaments, a winter sport that combines cross-country skiing races and rifle shooting. Each tournament consists of a series of rounds.

In each round, athletes compete on teams. In a biathlon, each athlete first skis a short distance, then shoots at a target.

Athletes are scored based on how fast they ski and how accurately they shoot. An athlete’s ability to ski is based on their intrinsic strength. Note that an athlete’s intrinsic strength remains constant throughout a tournament. An athlete neither gets stronger nor weaker between rounds. You can assume that all matches take place on the same day.

An athlete’s ability to shoot accurately is unrelated to their intrinsic strength. An athlete who is intrinsically strong may be a very good shooter, a very bad shooter, or something in between. Athlete’s shooting accuracy in any given round is also somewhat down to chance. 

A team’s overall score is determined by the sum of the scores for each athlete per event. Ski performance and shooting accuracy are weighted equally. Athletes receive more points for skiing faster and more points for achieving a high shooting accuracy.

The team that completes the course with the highest score wins. 
""",
    "skill": "strength",
    "latent": "shooting-accuracy",
    "match": "round"},    
}

latents = {
    "effort": {"description": "tried hard in", "preface": "Did", "question": 
    "On a percentage scale from 0 to 100%, how much effort do you think {player} put into the {match_idx} {match_token}?", "token": "effort"},
    "well-synchronized": {"description": "was well synchronized in", "preface": "Was", "question": "On a percentage scale from 0 to 100%, how well synchronized do you think {player} was with their teammate in the {match_idx} {match_token}?", "token": "well synchronized"},
    "difficulty": {"description": "attempt a difficult dive in", "preface": "Did", "question": "On a percentage scale from 0 to 100% (where 0=extremely easy, 100=as difficult as possible), how difficult of a dive do you think the team {player} was on in {match_idx} {match_token} attempted?", "token": "difficulty"},
    "shooting-accuracy": {"description": "had good shooting accuracy in", "preface": "Had", "question": "On a percentage scale from 0 to 100%, how accurate do you think {player} was at shooting in the {match_idx} {match_token}?", "token": "accurate"},
}

skill_map = {"skill": "more skilled", "strength": "stronger", "fast": "faster"}
skill_variable_map = {"skill" : "skill", "strength" : "strength", "fast" : "speed"}

sports_to_latent_variables_parses = {
    "tug-of-war": {
        "strength" : "intrinsic_strength_rank({{athlete: '{athlete}', out_of_n_athletes: 100}})",
        "effort" : "effort_percentage_level_in_match({{athlete: '{athlete}', match: {match}}})",
        "new_match" : "who_would_win_by_how_much({{team1: {team1}, team2: {team2}, match: {match}}})"
    },
    "canoe-race": {
        "strength" : "intrinsic_strength_rank({{athlete: '{athlete}', out_of_n_athletes: 100}})",
        "effort" : "effort_percentage_level_in_race({{athlete: '{athlete}', race: {match}}})",
        "new_match" : "who_would_win_by_how_much({{team1: {team1}, team2: {team2}, race: {match}}})"
    },
    "diving": {
        "skill" : "intrinsic_skill_rank({{athlete: '{athlete}', out_of_n_athletes: 100}})",
        "difficulty" : "difficulty_level_in_round({{team: '{team}', round: {match}}})",

        "new_match" : "who_would_win_by_how_much({{team1: {team1}, team2: {team2}, round: {match}}})"
    },
    "biathalon": {
        "strength" : "intrinsic_strength_rank({{athlete: '{athlete}', out_of_n_athletes: 100}})",
        "shooting-accuracy" : "accuracy_at_shooting_in_round({{athlete: '{athlete}', round: {match}}})",
        "new_match" : "who_would_win_by_how_much({{team1: {team1}, team2: {team2}, round: {match}}})"
    }
}