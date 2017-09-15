from sklearn import tree

BUMPY_SKIN = "bumpy"
SMOOTH_SKIN = "smooth"

SKINS_MAP = [BUMPY_SKIN, SMOOTH_SKIN]
BUMPY_SKIN_KEY = SKINS_MAP.index(BUMPY_SKIN)
SMOOTH_SKIN_KEY = SKINS_MAP.index(SMOOTH_SKIN)

APPLE = "apple"
ORANGE = "orange"

FRUIT_MAP = [APPLE, ORANGE]
APPLE_KEY = FRUIT_MAP.index(APPLE)
ORANGE_KEY = FRUIT_MAP.index(ORANGE)

features = [
    [140, SMOOTH_SKIN_KEY],
    [130, SMOOTH_SKIN_KEY],
    [150, BUMPY_SKIN_KEY],
    [170, BUMPY_SKIN_KEY],
]

labels = [
    APPLE_KEY,
    APPLE_KEY,
    ORANGE_KEY,
    ORANGE_KEY,
]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(features, labels)

guess_fruit_key = classifier.predict([[150, 0]])
print FRUIT_MAP[guess_fruit_key[0]]