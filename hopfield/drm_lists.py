"""DRM (Deese-Roediger-McDermott) word lists.

Each entry maps a critical lure (never studied) to its 15 studied associates,
ordered from strongest to weakest associate as in Roediger & McDermott (1995)
and Stadler, Roediger & McDermott (1999).  The lists were transcribed by hand
from those publications; proper nouns are lower-cased to match GloVe.

Human reference values used in the report (Stadler et al., 1999, recall):
  * window : 0.65 (highest of 36 lists)
  * king   : 0.10 (lowest of 36 lists)
  * mean over 36 lists : 0.46 (top half 0.51, bottom half 0.29)
"""

DRM_LISTS = {
    "sleep": ["bed", "rest", "awake", "tired", "dream", "wake", "snooze", "blanket",
              "doze", "slumber", "snore", "nap", "peace", "yawn", "drowsy"],
    "chair": ["table", "sit", "legs", "seat", "couch", "desk", "recliner", "sofa",
              "wood", "cushion", "swivel", "stool", "sitting", "rocking", "bench"],
    "doctor": ["nurse", "sick", "lawyer", "medicine", "health", "hospital", "dentist",
               "physician", "ill", "patient", "office", "stethoscope", "surgeon",
               "clinic", "cure"],
    "sweet": ["sour", "candy", "sugar", "bitter", "good", "taste", "tooth", "nice",
              "honey", "soda", "chocolate", "heart", "cake", "tart", "pie"],
    "mountain": ["hill", "valley", "climb", "summit", "top", "molehill", "peak",
                 "plain", "glacier", "goat", "bike", "climber", "range", "steep", "ski"],
    "needle": ["thread", "pin", "eye", "sewing", "sharp", "point", "prick", "thimble",
               "haystack", "thorn", "hurt", "injection", "syringe", "cloth", "knitting"],
    "rough": ["smooth", "bumpy", "road", "tough", "sandpaper", "jagged", "ready",
              "coarse", "uneven", "riders", "rugged", "sand", "boards", "ground", "gravel"],
    "slow": ["fast", "lethargic", "stop", "listless", "snail", "cautious", "delay",
             "traffic", "turtle", "hesitant", "speed", "quick", "sluggish", "wait",
             "molasses"],
    "river": ["water", "stream", "lake", "mississippi", "boat", "tide", "swim", "flow",
              "run", "barge", "creek", "brook", "fish", "bridge", "winding"],
    "window": ["door", "glass", "pane", "shade", "ledge", "sill", "house", "open",
               "curtain", "frame", "view", "breeze", "sash", "screen", "shutter"],
    "soft": ["hard", "light", "pillow", "plush", "loud", "cotton", "fur", "touch",
             "fluffy", "feather", "furry", "downy", "kitten", "skin", "tender"],
    "cold": ["hot", "snow", "warm", "winter", "ice", "wet", "frigid", "chilly", "heat",
             "weather", "freeze", "air", "shiver", "arctic", "frost"],
    "music": ["note", "sound", "piano", "sing", "radio", "band", "melody", "horn",
              "concert", "instrument", "symphony", "jazz", "orchestra", "art", "rhythm"],
    "man": ["woman", "husband", "uncle", "lady", "mouse", "male", "father", "strong",
            "friend", "beard", "person", "handsome", "muscle", "suit", "old"],
    "king": ["queen", "england", "crown", "prince", "george", "dictator", "palace",
             "throne", "chess", "rule", "subjects", "monarch", "royal", "leader", "reign"],
    "bread": ["butter", "food", "eat", "sandwich", "rye", "jam", "milk", "flour", "jelly",
              "dough", "crust", "slice", "wine", "loaf", "toast"],
    "spider": ["web", "insect", "bug", "fright", "fly", "arachnid", "crawl", "tarantula",
               "poison", "bite", "creepy", "animal", "ugly", "feelers", "small"],
    "thief": ["steal", "robber", "crook", "burglar", "money", "cop", "bad", "rob", "jail",
              "gun", "villain", "crime", "bank", "bandit", "criminal"],
    "foot": ["shoe", "hand", "toe", "kick", "sandals", "soccer", "yard", "walk", "ankle",
             "arm", "boot", "inch", "sock", "smell", "mouth"],
    "girl": ["boy", "dolls", "female", "young", "dress", "pretty", "hair", "niece",
             "dance", "beautiful", "cute", "date", "aunt", "daughter", "sister"],
    "high": ["low", "clouds", "up", "tall", "tower", "jump", "above", "building", "noon",
             "cliff", "sky", "over", "airplane", "dive", "elevate"],
    "anger": ["mad", "fear", "hate", "rage", "temper", "fury", "ire", "wrath", "happy",
              "fight", "hatred", "mean", "calm", "emotion", "enrage"],
    "black": ["white", "dark", "cat", "charred", "night", "funeral", "color", "grief",
              "blue", "death", "ink", "bottom", "coal", "brown", "gray"],
    "cup": ["mug", "saucer", "tea", "measuring", "coaster", "lid", "handle", "coffee",
            "straw", "goblet", "soup", "stein", "drink", "plastic", "sip"],
    "fruit": ["apple", "vegetable", "orange", "kiwi", "citrus", "ripe", "pear", "banana",
              "berry", "cherry", "basket", "juice", "salad", "bowl", "cocktail"],
    "smell": ["nose", "breathe", "sniff", "aroma", "hear", "see", "nostril", "whiff",
              "scent", "reek", "stench", "fragrance", "perfume", "salts", "rose"],
    "shirt": ["blouse", "sleeves", "pants", "tie", "button", "shorts", "iron", "polo",
              "collar", "vest", "pocket", "jersey", "belt", "linen", "cuffs"],
    "trash": ["garbage", "waste", "can", "refuse", "sewage", "bag", "junk", "rubbish",
              "sweep", "scraps", "pile", "dump", "landfill", "debris", "litter"],
    "car": ["truck", "bus", "train", "automobile", "vehicle", "drive", "jeep", "ford",
            "race", "keys", "garage", "highway", "sedan", "van", "taxi"],
    "lion": ["tiger", "circus", "jungle", "tamer", "den", "cub", "africa", "mane", "cage",
             "feline", "roar", "fierce", "bears", "hunt", "pride"],
    "pen": ["pencil", "write", "fountain", "leak", "quill", "felt", "bic", "scribble",
            "crayon", "cross", "tip", "marker", "red", "cap", "letter"],
}

# Published human false-recall probabilities quoted in the report.
HUMAN_FALSE_RECALL = {"window": 0.65, "king": 0.10}
HUMAN_MEAN_FALSE_RECALL = 0.46
