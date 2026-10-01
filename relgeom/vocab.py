"""Word pools for all materials. Every list is authored text; word sets are drawn from them reproducibly.

Names are invented roster entries; the name classes are material labels only. Contents are mostly multiword
dishes. The two-fact word sets of App. B are fixed literals (72 pairs of names, four foods each).
"""


def _split(text):
    return [item.strip() for item in text.replace("\n", "").split("|") if item.strip()]


FEMININE = _split("""
Ada|Adele|Agatha|Alma|Amelia|Annette|Ariana|Audrey|Aurora|Beatrix|Belinda|Bettina|Blanche|Brigitte|Carla|
Carmen|Catalina|Christina|Claudia|Constance|Cordelia|Corinne|Danielle|Dolores|Dora|Edwina|Eliza|Eloise|Elsa|
Emmeline|Eugenia|Eva|Frida|Gertrude|Hannelore|Heidi|Henrietta|Hilda|Ida|Ilse|Ines|Jennifer|Johanna|Julianne|
Katrina|Kristin|Larissa|Leonora|Liesel|Linnea|Lorena|Louisa|Lucille|Lydia|Mabel|Magdalena|Marcella|Marianne|
Marina|Marta|Maxine|Melissa|Mildred|Minerva|Monica|Muriel|Nadine|Nell|Nina|Nora|Octavia|Olga|Paloma|Philippa|
Priscilla|Ramona|Regina|Renata|Rhoda|Rosalie|Selma|Simone|Sofia|Susanna|Tabitha|Tamara|Tatiana|Thea|Ursula|
Valentina|Vanessa|Vera|Wilhelmina|Winifred|Yolanda|Zelda|Zora|Agnieszka|Birgit|Dagmar|Elfriede|Gudrun|
Hortense|Leocadia|Marisol|Ottilie|Perpetua|Rosamund|Solveig|Wanda|Xenia|Abigail|Adelaide|Adriana|Agnes|Aisha|
Alessandra|Alexandra|Alina|Amara|Anastasia|Anika|Annabel|Antonia|Ariadne|Astrid|Beatrice|Bianca|Bridget|
Camila|Cassandra|Cecilia|Celeste|Clara|Colette|Cora|Daphne|Delilah|Dorothea|Edith|Eleanor|Elena|Elisa|Elodie|
Emilia|Esme|Estelle|Evelyn|Fatima|Flora|Francesca|Freya|Gabriela|Genevieve|Georgia|Giselle|Greta|Gwendolyn|
Harriet|Hazel|Imogen|Ingrid|Isadora|Josephine|Judith|Juliet|Laila|Lavinia|Leila|Lenora|Lucia|Madeleine|Maeve|
Margot|Matilda|Miriam|Nadia|Naomi|Natalia|Ophelia|Penelope|Petra|Phoebe|Rosalind|Sabrina|Serena|Sylvia|
Theresa|Alberta|Andrea|Arlene|Barbara|Bernice|Beverly|Bonnie|Brianna|Candice|Cheryl|Connie|Cynthia|Delia|
Donna|Doris|Edna|Ella|Elsie|Emily|Erica|Ethel|Fern|Georgette|Geraldine|Gina|Glenda|Hattie|Hope|Inez|Irma|Ivy|
Jane|Janice|Jasmine|Jill|Joan|Josie|Kara|Kimberly|Kristen|Leah|Lena|Leona|Lila|Lois|Loretta|Marcia|Marilyn|
Melanie|Minnie|Mona|Myra|Olive|Opal|Paula|Pearl|Renee|Rita|Roberta|Rosemary|Roxanne|Sally|Sandra|Shirley|
Sonia|Stella|Tara|Thelma|Tracy|Viola|Wilma|Dahlia|Eunice|Hester|Imelda|Juanita|Lorna|Rosa|Sophia|Verity|
Yasmin|Zara|Alicia|Carolyn|Felicia|Jolene|Mavis|Odette|Rowena|Sadie|Trudy|
""")

MASCULINE = _split("""
Abraham|Alfonso|Alonzo|Andre|Angus|Archibald|Armando|Augustin|Barnaby|Bartholomew|Bertram|Bruno|Byron|Calvin|
Carlos|Casimir|Cecil|Cornelius|Dante|Dmitri|Eamon|Edgar|Edwin|Emmett|Enrique|Ezra|Fabian|Ferdinand|Fergus|
Gabriel|Gilbert|Giovanni|Gustav|Harvey|Herbert|Horace|Ignatius|Jacob|Joaquin|Joseph|Joshua|Klaus|Leon|Leopold|
Lionel|Lucas|Ludwig|Luther|Magnus|Manuel|Marco|Maurice|Maximilian|Mordecai|Neville|Octavio|Orlando|Oswald|
Pablo|Percival|Roderick|Roland|Rudolph|Rupert|Silas|Solomon|Stanley|Stefan|Thaddeus|Tobias|Ulysses|Vernon|
Wendell|Wilfred|Xavier|Zachary|Alvaro|Anselm|Benedikt|Cosimo|Detlef|Emilio|Florian|Gaspard|Gottfried|Hamish|
Horatio|Ingmar|Jarvis|Lars|Leander|Mortimer|Nikolai|Olaf|Osvaldo|Rainer|Rufus|Sigmund|Torsten|Ulrich|Valentin|
Waldemar|Wolfgang|Yusuf|Zoltan|Ambrosio|Benno|Crispin|Evander|Aaron|Abel|Adrian|Alastair|Albert|Alden|Alfred|
Ambrose|Amir|Ansel|Anton|Arthur|August|Benjamin|Bernard|Blake|Boris|Caleb|Callum|Cedric|Charles|Clarence|
Clement|Colin|Conrad|Cyrus|Damian|Daniel|Darius|Declan|Desmond|Dominic|Douglas|Edmund|Edward|Elias|Emil|Ethan|
Felix|Frederick|Gareth|Gideon|Graham|Hector|Hugo|Isaac|Isaiah|Jasper|Jerome|Jonah|Jonathan|Julian|Laurence|
Leonard|Lorenzo|Louis|Lucian|Malcolm|Marcus|Martin|Matteo|Maxwell|Miles|Nathan|Nicolas|Oliver|Oscar|Otis|
Patrick|Peter|Philip|Quentin|Raphael|Sebastian|Theodore|Alvin|Barry|Chester|Clayton|Clive|Dale|Dexter|Dudley|
Earl|Elmer|Everett|Floyd|Forrest|Garrett|Glenn|Grant|Ivor|Jerry|Joel|Julius|Lance|Leroy|Lloyd|Lyle|Marshall|
Melvin|Milton|Murray|Noel|Orson|Owen|Percy|Quincy|Ralph|Rex|Rodney|Seymour|Sidney|Stuart|Terence|Travis|
Tristan|Virgil|Wallace|Wesley|Wilbur|Willard|Winston|Barrett|Jonas|Kurt|Orville|Pierce|Sterling|Thornton|Zeke|
Ahmed|Cyril|Felipe|Gregor|Hans|Ibrahim|Jorge|Kofi|Nestor|Omar|Rafael|Santiago|Tomas|Viktor|Yuri|Rolf|Dorian|
Esteban|Lazarus|Reinhold|
""")

NEUTRAL = _split("""
Alex|Ainsley|Bellamy|Blair|Bobbie|Carson|Corey|Dana|Darby|Emery|Harley|Jaden|Jody|Justice|Kai|Kerry|Kit|
Leslie|Lindsay|Loren|Lou|Mackenzie|Marion|Max|Micah|Nico|Ollie|Pat|Perry|Presley|Raleigh|Ray|Reagan|Rory|
Sasha|Scout|Shannon|Shay|Stevie|Tatum|Toby|Val|Whitney|Zion|Brett|Cody|Dylan|Erin|Jaime|Jan|Kay|Lennon|London|
Lyric|Marlowe|Merritt|Nova|Phoenix|Rene|Rio|Rylan|Sawyer|Sloane|Tanner|Tory|Vesper|Winslow|Ashby|Campbell|
Delaney|Ellison|Greer|Addison|Alexis|Arden|Ariel|Aubrey|Avery|Bailey|Briar|Brook|Cameron|Casey|Dakota|Dallas|
Devon|Drew|Eden|Ellery|Emerson|Finley|Frankie|Gray|Harper|Hayden|Hollis|Jamie|Jesse|Jordan|Jules|Kendall|
Kennedy|Lane|Logan|Marley|Morgan|Oakley|Parker|Peyton|Quinn|Reese|Remy|Riley|River|Robin|Rowan|Sage|Shiloh|
Skyler|Spencer|Taylor|Teagan|Wren|
""")

SURNAMES = _split("""
Abernathy|Aldridge|Ashford|Atwood|Bancroft|Barlow|Beckett|Blackwell|Bradshaw|Brennan|Burgess|Calloway|Carver|
Chandler|Colby|Cromwell|Dalrymple|Davenport|Dempsey|Donovan|Dunmore|Eastwood|Ellsworth|Fairbanks|Farrell|
Fenwick|Fitzgerald|Forsythe|Galloway|Garland|Gilmore|Goodwin|Granger|Hadley|Halloran|Hargrove|Hartley|
Hawthorne|Holloway|Huxley|Ingersoll|Kavanagh|Keating|Kingsley|Kirkwood|Langford|Larkin|Lockhart|Lombard|
Lowell|Maddox|Mansfield|Merriweather|Montague|Mulligan|Northcott|Oakes|Ogilvie|Pembroke|Pendleton|Prescott|
Radcliffe|Redford|Rowntree|Rutherford|Sackville|Sheffield|Sinclair|Stanhope|Stratton|Sullivan|Sutherland|
Talbot|Thackeray|Thornbury|Trevelyan|Underhill|Upton|Vandermeer|Wainwright|Wakefield|Waverly|Weatherby|
Whitaker|Whitmore|Winslow|Woodward|Wyndham|Yardley|Blythe|Corrigan|Delaney|Everly|Faulkner|Gallagher|Hennessy|
Iverson|Kendrick|Lindqvist|Mercer|Nolan|Pritchard|Quigley|Rafferty|Sorensen|Tennyson|Vickers|Whitfield|
Ashcroft|Bellamy|Castellano|Draycott|Fairfax|Grimshaw|Holbrook|Kimball|Lancaster|Marchetti|Novak|Okafor|
Petrov|Rinaldi|Castillo|Moreno|Nakamura|Tanaka|Kowalski|Lindgren|Haddad|Mendoza|Rossi|Schneider|Varga|Weber|
""")

# Foods, dishes first; no two share a normalized string.
FOODS = _split("""
apple turnover|blackberry pie|cranberry scone|date loaf|beef brisket|lamb shank|pork belly|veal cutlet|
fish tacos|shrimp scampi|lobster roll|clam linguine|eggs benedict|breakfast burrito|home fries|sausage gravy|
onion rings|mozzarella sticks|potato wedges|garlic knots|nicoise salad|chef salad|taco salad|wedge salad|
lemon tart|banana bread|vanilla custard|fig pudding|beef jerky|trail mix|granola bar|rice crackers|
tomato bisque|lentil soup|mushroom barley soup|chicken noodle soup|fried rice|egg drop soup|orange chicken|
lo mein|chicken korma|paneer tikka|vegetable biryani|malai kofta|sesame noodles|pork buns|dan dan noodles|
xiaolongbao|toad in the hole|yorkshire pudding|steak pie|beef wellington|corned beef|pastrami sandwich|
matzo ball soup|knish|grilled cheese|tuna melt|club sandwich|monte cristo|glazed ham|pot roast|roast goose|
stuffed turkey|pad see ew|green curry|mango sticky rice|tom kha gai|kimchi stew|galbi|haemul pajeon|
doenjang jjigae|chicken katsu|beef teriyaki|salmon nigiri|miso ramen|carne asada|mole poblano|elote|
pork carnitas|black bean soup|chili con carne|nachos|fajitas|pierogi|kielbasa|beet salad|cabbage rolls|
pelmeni|kasha|blini|solyanka|coq au vin|beef bourguignon|duck confit|onion tart|lemon meringue pie|
apple strudel|pecan tart|coconut cream pie|banana split|hot fudge sundae|milkshake|ice cream sandwich|
oatmeal raisin cookies|shortbread|gingerbread|fig newtons|cheesecake|pound cake|angel food cake|
red velvet cake|blueberry muffins|danish pastry|sticky buns|cream puffs|peanut butter sandwich|ham and cheese|
egg salad sandwich|veggie wrap|cranberry sauce|stuffing|green bean casserole|yam casserole|baked beans|
coleslaw|pulled pork|smoked brisket|buffalo wings|jalapeno poppers|potato skins|deviled eggs|
spaghetti bolognese|linguine with clams|fettuccine alfredo|orecchiette|margherita pizza|calzone|stromboli|
white pizza|osso buco|chicken parmesan|veal piccata|eggplant parmesan|arancini|bruschetta|prosciutto|
caprese salad|gyro|souvlaki|dolmades|pastitsio|kofta|beef shawarma|mujadara|lamb ouzi|jollof rice|fufu|
egusi soup|suya|bobotie|peri peri chicken|milk tart|chakalaka|doro wat|kitfo|tibs|shiro|ceviche|lomo saltado|
aji de gallina|papa rellena|feijoada|pao de queijo|coxinha|brigadeiro|jerk chicken|rice and peas|
ackee and saltfish|curry goat|poutine|butter tart|nanaimo bars|tourtiere|meat pie|lamington|pavlova|
sausage roll|roti canai|char kway teow|hainanese chicken rice|otak otak|banh mi|com tam|goi cuon|hu tieu|
sinigang|lechon|pancit|kare kare|twice cooked pork|salt and pepper squid|egg tarts|beef ho fun|
tandoori chicken|rogan josh|saag aloo|chicken jalfrezi|vegetable pulao|rajma|kheer|gulab jamun|mushroom soup|
vichyssoise|chicken broth|tortilla soup|turkey club|philly cheesesteak|sloppy joe|meatball sub|cheeseburger|
hot dog|fried chicken|veggie burger|glazed donuts|apple fritters|churros|beignets|key lime pie|
sweet potato pie|peach cobbler|chess pie|rocky road|peach melba|fudge brownies|lemon bars|trifle|eton mess|
jam roly poly|bakewell tart|deep dish pizza|hawaiian pizza|sicilian pizza|flatbread pizza|steak frites|
filet mignon|prime rib|beef tartare|seared tuna|grilled swordfish|baked cod|poached salmon|shrimp gumbo|
crawfish boil|hush puppies|dirty rice|biscuits and gravy|meatloaf|fried okra|chicken fried steak|lentil stew|
vegetable curry|mushroom risotto|stuffed zucchini|tuna casserole|scalloped potatoes|ambrosia salad|
gelatin salad|popcorn|cotton candy|funnel cake|kettle corn|yogurt parfait|acai bowl|smoothie bowl|
chia pudding|overnight oats|bran flakes|corn flakes|muesli|goat cheese|blue cheese|swiss cheese|cheddar|
smoked salmon|pickled herring|gravlax|fish roe|cornish hen|quail eggs|duck breast|venison stew|crepes suzette|
madeleines|tarte tatin|financiers|chicken marsala|beef braciole|shrimp fra diavolo|lobster ravioli|
zucchini fritters|spanakopita|halloumi fries|taramasalata|almond croissant|baked alaska|cabbage soup|
duck ragu|egg fried rice|fish fingers|ham hock|irish stew|jam tart|kale chips|lemon chicken|meatball soup|
nut roast|oxtail stew|pea soup|quiche lorraine|rhubarb crumble|sausage casserole|tomato risotto|unagi don|
vegetable tagine|walnut bread|yakitori|ziti bake|chicken kiev|chicken piccata|veal marsala|pork chops|
lamb chops|chicken cacciatore|steak diane|pepper steak|salisbury steak|country fried steak|chicken fried rice|
pineapple fried rice|beef lo mein|vegetable lo mein|sesame chicken|general tso chicken|cashew chicken|
honey garlic chicken|lamb rogan josh|beef madras|prawn curry|fish curry|red curry|massaman curry|panang curry|
thai basil chicken|pork larb|beef pho|katsu curry|salmon teriyaki|miso salmon|grilled halibut|
pan seared scallops|garlic prawns|shrimp tacos|beef burrito|chicken quesadilla|bean burrito|pork tamales|
chicken enchiladas|beef empanadas|cheese enchiladas|chile relleno|chicken mole|arroz con pollo|ropa vieja|
black bean chili|turkey chili|white chicken chili|bacon cheeseburger|turkey burger|lamb burger|corn dog|
chili dog|reuben sandwich|cuban sandwich|penne vodka|seafood paella|chicken alfredo|pesto pasta|cacio e pepe|
baked rigatoni|beef lasagna|spinach lasagna|pumpkin ravioli|shrimp linguine|sweet potato casserole|
chicken casserole|beef noodle soup|pork adobo|beef bulgogi|kimchi fried rice|tonkotsu ramen|shoyu ramen|
tempura udon|cold soba|salmon sashimi|tuna sashimi|california roll|spicy tuna roll|dragon roll|
chicken yakitori|pork gyoza|shrimp dumplings|soup dumplings|xiao long bao|lamb skewers|mongolian beef|
crispy duck|roast duck|pad kra pao|green papaya salad|chicken laksa|chicken biryani|lamb biryani|
chicken vindaloo|masala dosa|vegetable samosa|chole bhature|rajma chawal|pav bhaji|vada pav|mutton curry|
lamb tagine|chicken tagine|beef kofta|chicken kebab|doner kebab|lamb gyro|chicken souvlaki|
stuffed grape leaves|spinach pie|cheese borek|wiener schnitzel|chicken schnitzel|potato pancakes|
potato dumplings|chicken fricassee|beef carpaccio|veal scallopini|chicken francese|risotto milanese|
pasta alla norma|polenta cakes|swedish meatballs|herring salad|cheese pierogi|potato pierogi|
lancashire hotpot|beef and ale pie|bubble and squeak|eggs florentine|spanish omelette|patatas bravas|
gambas al ajillo|chicken croquettes|ham croquettes|croque monsieur|croque madame|salade nicoise|tuna tartare|
moules frites|sole meuniere|chicken chasseur|baked potato|twice baked potatoes|curly fries|cheese fries|
creamed spinach|glazed carrots|roasted brussels sprouts|grilled asparagus|corn on the cob|creamy coleslaw|
fried green tomatoes|coconut rice|fried plantains|yellow rice|wild rice pilaf|herb stuffing|garden salad|
spinach salad|pasta bake|three bean salad|quinoa salad|couscous salad|cucumber salad|kale salad|chicken salad|
tuna salad|shrimp salad|crab salad|macaroni salad|buttermilk biscuits|steel cut oats|granola parfait|
avocado toast|smoked salmon bagel|belgian waffles|buckwheat crepes|lemon crepes|chocolate crepes|
strawberry crepes|cheese omelette|western omelette|denver omelette|scrambled eggs|poached eggs|fried eggs|
corned beef hash|breakfast sausage|apple pie|cherry pie|banana cream pie|bran muffins|chocolate cake|
lemon drizzle cake|fruit cake|new york cheesecake|chocolate fudge|bread and butter pudding|chocolate pudding|
cherry strudel|sugar cookies|shortbread cookies|almond biscotti|chocolate eclairs|jelly doughnuts|
rice krispie treats|chocolate truffles|butterscotch pudding|caramel flan|tres leches cake|
churros con chocolate|strawberry sundae|pistachio ice cream|coconut ice cream|lemon sorbet|matcha ice cream|
mochi ice cream|red bean buns|sesame balls|almond jelly|loaded nachos|chicken wings|stuffed mushrooms|
pigs in blankets|sausage rolls|cheese straws|tomato bruschetta|pretzel bites|potato chips|tortilla chips|
spinach dip|french fries|sweet corn fritters|vegetable tempura|shrimp toast|crab rangoon|scallion pancakes|
chicken nuggets|popcorn chicken|chicken tenders|butternut squash soup|leek and potato soup|carrot ginger soup|
beef barley soup|oxtail soup|pumpkin soup|cream of mushroom|fish chowder|seafood gumbo|italian wedding soup|
onion bhaji|tomato soup|minestrone soup|vegetable stew|chicken stew|lamb stew|roast turkey|turkey sandwich|
chicken pesto sandwich|egg and cress sandwich|fluffy pancakes|blueberry scones|cinnamon toast|honey granola|
apricot jam tart|plum crumble|gooseberry fool|summer pudding|spotted dick|mince pies|christmas pudding|
carrot soup|garlic soup|nettle soup|rabbit stew|goose fat potatoes|pheasant pie|game pie|pork pie|
beef casserole|chicken pie|mushroom pie|vegetable pie|lentil shepherds pie|stuffed squash|halloumi skewers|
grilled vegetables|roasted peppers|ratatouille tart|leek tart|goat cheese tart|spinach quiche|quinoa|couscous|
bulgur|millet|buckwheat|amaranth|sorghum|teff|barley|farro|freekeh|spelt|oatmeal|polenta|grits|congee|risotto|
paella|jambalaya|pilaf|biryani|kedgeree|bibimbap|nasi goreng|sushi|onigiri|kimbap|chirashi|ramen|udon|soba|
somen|pho|laksa|pad thai|chow mein|lasagna|ravioli|gnocchi|tortellini|fettuccine|tagliatelle|orzo|vermicelli|
falafel|hummus|tabbouleh|fattoush|baba ganoush|muhammara|labneh|tzatziki|ratatouille|moussaka|shakshuka|
caponata|gazpacho|minestrone|borscht|goulash|miso soup|tom yum|harira|pozole|taco|burrito|enchilada|
quesadilla|tamale|arepa|pupusa|empanada|samosa|pakora|bhaji|kachori|dosa|idli|uttapam|dhokla|naan|roti|
paratha|chapati|bagel|croissant|brioche|focaccia|ciabatta|pita|injera|lavash|pretzel|scone|crumpet|bannock|
frittata|omelet|quiche|souffle|tofu|tempeh|seitan|edamame|aubergine|artichoke|asparagus|okra|zucchini|
cauliflower|kohlrabi|fennel|endive|escarole|watercress|arugula|chard|kale|collard greens|bok choy|parsnip|
rutabaga|daikon|salsify|taro|cassava|jicama|plantain|shiitake|chanterelle|portobello|enoki|persimmon|
pomegranate|guava|papaya|lychee|rambutan|longan|mangosteen|durian|jackfruit|soursop|cherimoya|starfruit|
passion fruit|dragon fruit|breadfruit|blackberry|raspberry|blueberry|gooseberry|cranberry|mulberry|
boysenberry|elderberry|nectarine|quince|loquat|kumquat|clementine|tangerine|pomelo|yuzu|pistachio|pecan|
hazelnut|macadamia|chestnut|pine nut|brazil nut|ginkgo nut|sesame seeds|sunflower seeds|pumpkin seeds|poppy seeds|black eyed peas|kidney beans|mung beans|lima beans|
halibut|haddock|mackerel|sardine|anchovy|herring|tilapia|swordfish|scallop|mussel|clam|abalone|calamari|
octopus|cuttlefish|sea urchin|baklava|tiramisu|panna cotta|cannoli|strudel|eclair|profiterole|macaron|halva|
nougat|marzipan|turkish delight|chicken soup|beef stew|fish pie|lentil curry|potato salad|pasta salad|
fruit salad|egg salad|roast chicken|roast beef|roast pork|roast lamb|grilled salmon|baked trout|
smoked haddock|fried cod|apple crumble|pear tart|cherry cobbler|peach crisp|chocolate brownie|
lemon cheesecake|carrot cake|coffee cake|rice pudding|bread pudding|banana pudding|tapioca pudding|
garlic bread|cornbread|sourdough toast|cheese scone|chicken sandwich|tuna sandwich|egg sandwich|ham sandwich|
cheese pizza|mushroom pizza|pepperoni pizza|seafood pizza|spaghetti carbonara|penne arrabbiata|
linguine alle vongole|pasta primavera|macaroni and cheese|baked ziti|stuffed shells|pasta puttanesca|
chicken tikka|butter chicken|lamb vindaloo|chana masala|palak paneer|dal makhani|aloo gobi|baingan bharta|
beef rendang|chicken adobo|pork sisig|fish amok|chicken satay|beef satay|nasi lemak|mee goreng|spring rolls|
egg rolls|wonton soup|hot and sour soup|kung pao chicken|mapo tofu|sweet and sour pork|beef chow fun|char siu|
peking duck|soy sauce chicken|steamed pork buns|chicken teriyaki|pork tonkatsu|beef sukiyaki|shrimp tempura|
gyoza|okonomiyaki|takoyaki|yakisoba|tteokbokki|japchae|bulgogi|samgyetang|beef stroganoff|chicken paprikash|
pork schnitzel|stuffed cabbage|shepherds pie|cottage pie|steak and kidney pie|cornish pasty|fish and chips|
bangers and mash|scotch egg|welsh rarebit|french onion soup|lobster bisque|clam chowder|split pea soup|
crab cakes|fish cakes|salmon patties|shrimp cocktail|chicken pot pie|turkey meatloaf|beef meatballs|
pork dumplings|falafel wrap|hummus wrap|chicken shawarma|lamb kebab|greek salad|caesar salad|waldorf salad|
cobb salad|stuffed peppers|baked eggplant|roasted cauliflower|sauteed mushrooms|mashed potatoes|potato gratin|
sweet potato fries|hash browns|blueberry pancakes|banana waffles|french toast|cinnamon rolls|
sticky toffee pudding|creme brulee|chocolate mousse|lemon posset|strawberry shortcake|black forest cake|
victoria sponge|pineapple upside down cake|chocolate chip cookies|oatmeal cookies|ginger snaps|
peanut butter cookies|custard tart|treacle tart|pecan pie|pumpkin pie|vanilla ice cream|mango sorbet|
raspberry gelato|frozen custard|beef chili|chicken fajitas|carnitas tacos|huevos rancheros|red beans and rice|
shrimp and grits|chicken gumbo|corn chowder|
""")

PETS = _split("""
dog|cat|rabbit|hamster|parrot|goldfish|turtle|snake|lizard|ferret|horse|pony|donkey|goat|sheep|pig|cow|duck|
goose|canary|cockatoo|iguana|gecko|chameleon|tarantula|scorpion|hedgehog|chinchilla|gerbil|guinea pig|mouse|
rat|frog|toad|salamander|axolotl|llama|alpaca|camel|peacock|owl|falcon|hawk|pigeon|dove|raven|swan|ostrich|
emu|tortoise|crab|lobster|monkey|raccoon|skunk|squirrel|chipmunk|beaver|otter|bat|sloth|lemur|meerkat|python|
cobra|alligator|crocodile|zebra|kangaroo|penguin|flamingo|aardvark|anteater|antelope|armadillo|badger|bear|
beetle|bison|boar|bobcat|bull|butterfly|capybara|caribou|cheetah|chicken|chimpanzee|cougar|coyote|crane|crow|
deer|dingo|dolphin|dragonfly|eagle|eel|elephant|elk|firefly|gazelle|gibbon|giraffe|gopher|gorilla|grasshopper|
grouse|hare|heron|hippo|hornet|hummingbird|hyena|ibex|jackal|jaguar|jellyfish|koala|ladybug|leopard|lion|lynx|
macaw|magpie|manatee|marmot|mink|mole|mongoose|moose|moth|mule|narwhal|newt|ocelot|octopus|opossum|orangutan|
oriole|oyster|panda|panther|parakeet|partridge|pelican|pheasant|platypus|porcupine|puffin|quail|reindeer|
rhino|robin|rooster|sardine|seahorse|seal|shark|snail|sparrow|spider|squid|starfish|stingray|stork|swordfish|
tapir|termite|toucan|vulture|walrus|warthog|wasp|weasel|whale|wolf|wombat|woodpecker|yak|budgie|cockatiel|
lovebird|mynah|guppy|angelfish|koi|tetra|cichlid|catfish|carp|piranha|clownfish|pufferfish|hermit crab|
tree frog|bullfrog|corn snake|king snake|boa|anaconda|viper|rattlesnake|monitor lizard|bearded dragon|skink|
komodo dragon|millipede|centipede|stick insect|praying mantis|silkworm|earthworm|leech|sugar glider|degu|
jerboa|dormouse|vole|shrew|pika|marten|wolverine|stoat|ermine|polecat|civet|genet|binturong|kinkajou|coati|
quokka|wallaby|bandicoot|echidna|numbat|dugong|orca|beluga|porpoise|manta ray|barracuda|marlin|lamprey|
nautilus|cuttlefish|crayfish|sea urchin|sea turtle|albatross|cormorant|egret|gannet|ibis|kingfisher|
kookaburra|loon|nightingale|osprey|petrel|plover|sandpiper|starling|swallow|tern|thrush|turkey|warbler|
cassowary|rhea|condor|buzzard|kestrel|zebu|water buffalo|okapi|vicuna|guanaco|mandrill|baboon|macaque|
marmoset|tamarin|capuchin|howler monkey|tortoiseshell cat|siamese cat|persian cat|border collie|beagle|poodle|
dachshund|labrador|bulldog|greyhound|chihuahua|terrier|spaniel|dalmatian|husky|rottweiler|pug|corgi|
shetland pony|guinea fowl|bantam|mallard|pigeon hawk|barn owl|snowy owl|black bear|polar bear|grizzly bear|
red panda|arctic fox|fennec fox|snow leopard|white rhino|sea lion|elephant seal|fruit bat|vampire bat|
tree kangaroo|flying squirrel|prairie dog|musk ox|mountain goat|bighorn sheep|wild boar|honey badger|
killer bee|bumblebee|monarch butterfly|
""")

CITIES = _split("""
Paris|London|Tokyo|Berlin|Madrid|Rome|Vienna|Prague|Dublin|Lisbon|Athens|Cairo|Moscow|Beijing|Shanghai|Seoul|
Bangkok|Sydney|Melbourne|Toronto|Chicago|Boston|Denver|Seattle|Houston|Miami|Dallas|Phoenix|Atlanta|Montreal|
Vancouver|Lima|Santiago|Havana|Nairobi|Lagos|Dakar|Accra|Casablanca|Istanbul|Dubai|Mumbai|Delhi|Kolkata|
Karachi|Tehran|Baghdad|Jakarta|Manila|Hanoi|Singapore|Kyoto|Osaka|Oslo|Stockholm|Helsinki|Copenhagen|Warsaw|
Budapest|Brussels|Amsterdam|Zurich|Geneva|Munich|Hamburg|Milan|Venice|Naples|Barcelona|Seville|Porto|
Edinburgh|Glasgow|Manchester|Liverpool|Marseille|Lyon|Krakow|Belgrade|Bucharest|Riga|Tallinn|Vilnius|Auckland|
Perth|Brisbane|Honolulu|Anchorage|Nashville|Memphis|Detroit|Portland|San Diego|Buenos Aires|Rio de Janeiro|
Quito|Caracas|Cape Town|Johannesburg|Addis Ababa|Kampala|Tunis|Algiers|Beirut|Amman|Doha|Riyadh|Lahore|Dhaka|
Kathmandu|Colombo|Taipei|Hong Kong|Kuala Lumpur|Montevideo|Minneapolis|Los Angeles|San Francisco|New York|
Philadelphia|Baltimore|Pittsburgh|Cleveland|Cincinnati|Milwaukee|Kansas City|Omaha|Tulsa|Albuquerque|Tucson|
Las Vegas|Sacramento|Salt Lake City|Boise|Spokane|Oakland|Fresno|San Antonio|El Paso|New Orleans|Orlando|
Tampa|Jacksonville|Charleston|Raleigh|Quebec City|Ottawa|Calgary|Edmonton|Winnipeg|Halifax|Mexico City|
Guadalajara|Monterrey|Tijuana|Cancun|Panama City|San Jose|Bogota|Medellin|Cartagena|La Paz|Asuncion|Cordoba|
Rosario|Mendoza|Valparaiso|Sao Paulo|Brasilia|Salvador|Recife|Fortaleza|Manaus|Guatemala City|Tegucigalpa|
Managua|Kingston|Nassau|San Juan|Santo Domingo|Reykjavik|Bergen|Gothenburg|Malmo|Aarhus|Tampere|Turku|
Rotterdam|Utrecht|Antwerp|Ghent|Bruges|Luxembourg|Cologne|Frankfurt|Stuttgart|Dresden|Leipzig|Hanover|Bremen|
Dusseldorf|Nuremberg|Heidelberg|Salzburg|Innsbruck|Graz|Basel|Bern|Lausanne|Lucerne|Turin|Genoa|Bologna|
Verona|Palermo|Pisa|Siena|Valencia|Bilbao|Malaga|Granada|Zaragoza|Toledo|Coimbra|Bordeaux|Toulouse|Strasbourg|
Lille|Nantes|Montpellier|Cannes|Monaco|Bristol|Leeds|Birmingham|Sheffield|Newcastle|Oxford|Cambridge|Cardiff|
Belfast|Galway|Aberdeen|Inverness|Gdansk|Wroclaw|Poznan|Brno|Bratislava|Ljubljana|Zagreb|Dubrovnik|Sarajevo|
Skopje|Plovdiv|Thessaloniki|Tirana|Chisinau|Kyiv|Odesa|Lviv|Minsk|Saint Petersburg|Novosibirsk|Kazan|
Vladivostok|Tbilisi|Yerevan|Baku|Almaty|Tashkent|Samarkand|Bishkek|Ulaanbaatar|Kabul|Islamabad|Peshawar|
Chennai|Bangalore|Hyderabad|Pune|Jaipur|Agra|Varanasi|Ahmedabad|Lucknow|Kochi|Chittagong|Thimphu|Yangon|
Mandalay|Phnom Penh|Vientiane|Chiang Mai|Phuket|Ho Chi Minh City|Da Nang|Shenzhen|Chengdu|Chongqing|Wuhan|
Nanjing|Tianjin|Harbin|Qingdao|Kunming|Lhasa|Macau|Busan|Incheon|Pyongyang|Sapporo|Nagoya|Yokohama|Kobe|
Hiroshima|Fukuoka|Nara|Kaohsiung|Cebu|Davao|Surabaya|Bandung|Yogyakarta|Medan|Adelaide|Canberra|Hobart|Cairns|
Christchurch|Queenstown|Suva|Port Moresby|Jerusalem|Tel Aviv|Damascus|Aleppo|Muscat|Abu Dhabi|Kuwait City|
Manama|Jeddah|Mecca|Isfahan|Shiraz|Tabriz|Ankara|Izmir|Antalya|Nicosia|Alexandria|Luxor|Aswan|Khartoum|
Tripoli|Benghazi|Marrakesh|Fez|Rabat|Tangier|Bamako|Niamey|Ouagadougou|Abidjan|Freetown|Monrovia|Conakry|
Abuja|Kano|Ibadan|Douala|Yaounde|Libreville|Kinshasa|Brazzaville|Luanda|Lusaka|Harare|Maputo|Gaborone|
Windhoek|Pretoria|Durban|Mombasa|Dar es Salaam|Zanzibar|Kigali|Mogadishu|Asmara|Antananarivo|Port Louis|
""")

JOBS = _split("""
teacher|nurse|doctor|lawyer|engineer|architect|pilot|chef|butcher|carpenter|plumber|electrician|mechanic|
farmer|fisherman|firefighter|police officer|soldier|sailor|dentist|pharmacist|surgeon|veterinarian|accountant|
banker|cashier|waiter|bartender|librarian|journalist|photographer|painter|sculptor|musician|singer|dancer|
actor|writer|poet|editor|translator|programmer|designer|tailor|barber|hairdresser|gardener|janitor|locksmith|
jeweler|potter|welder|miner|lumberjack|beekeeper|shepherd|judge|professor|scientist|chemist|physicist|
biologist|geologist|astronomer|economist|psychologist|therapist|paramedic|midwife|optician|receptionist|
secretary|clerk|courier|taxi driver|bus driver|truck driver|mail carrier|nanny|tutor|coach|referee|lifeguard|
zookeeper|florist|grocer|detective|diplomat|mayor|priest|surveyor|auctioneer|glassblower|blacksmith|
stonemason|actuary|agronomist|air traffic controller|anesthesiologist|animator|anthropologist|appraiser|
archaeologist|archivist|art director|astronaut|auditor|author|bailiff|bellhop|biochemist|bodyguard|bookbinder|
bookkeeper|botanist|bricklayer|stockbroker|builder|cabinetmaker|camera operator|cardiologist|caretaker|
cartographer|cartoonist|caterer|chauffeur|chiropractor|choreographer|civil servant|cleaner|clockmaker|cobbler|
columnist|comedian|composer|conductor|confectioner|consultant|cook|copywriter|coroner|counselor|
crane operator|critic|croupier|curator|customs officer|data analyst|deckhand|decorator|delivery driver|
dermatologist|dietitian|dishwasher|dispatcher|dockworker|dog walker|doorman|draftsman|ecologist|embalmer|
engraver|entomologist|entrepreneur|epidemiologist|estate agent|event planner|farmhand|fashion designer|
ferryman|film director|financial analyst|fishmonger|fitness instructor|flight attendant|forester|
forensic scientist|fortune teller|funeral director|furrier|game warden|gamekeeper|geneticist|glazier|
goldsmith|graphic designer|greengrocer|groundskeeper|gunsmith|harbor master|historian|horticulturist|
hotel manager|housekeeper|illustrator|immigration officer|innkeeper|insurance agent|interior designer|
interpreter|jockey|joiner|juggler|lab technician|landscaper|lecturer|legislator|lexicographer|linguist|
loan officer|lyricist|machinist|magician|marine biologist|marketing manager|mathematician|meteorologist|
microbiologist|milkman|fashion model|mortician|museum guide|neurologist|neurosurgeon|novelist|nutritionist|
obstetrician|oceanographer|oncologist|optometrist|orthodontist|paleontologist|paralegal|park ranger|pastor|
pathologist|pawnbroker|pediatrician|personal trainer|philosopher|physiotherapist|playwright|podiatrist|
politician|postmaster|prison guard|private investigator|film producer|proofreader|prosecutor|psychiatrist|
publicist|publisher|puppeteer|radio host|radiologist|rancher|recruiter|reporter|researcher|roofer|saddler|
salesperson|scaffolder|screenwriter|seamstress|security guard|senator|sheriff|shoemaker|shop assistant|
silversmith|social worker|software engineer|sommelier|songwriter|sound engineer|speech therapist|statistician|
street vendor|stuntman|systems analyst|tax collector|taxidermist|television presenter|tiler|toolmaker|
tour guide|town planner|toymaker|train driver|travel agent|tree surgeon|typist|upholsterer|usher|valet|
video editor|vintner|watchmaker|web developer|wedding planner|window cleaner|wine merchant|woodcarver|
youth worker|zoologist|auto mechanic|bus conductor|choir director|court reporter|crossing guard|
dental hygienist|drill sergeant|ferry captain|gallery owner|head chef|ice cream vendor|lighthouse keeper|
mail sorter|night watchman|office manager|pastry chef|radio operator|ship captain|sign painter|ski instructor|
stable hand|swimming instructor|tattoo artist|tennis coach|tugboat captain|vineyard worker|
""")

INSTRUMENTS = _split("""
piano|violin|cello|viola|guitar|harp|flute|clarinet|oboe|bassoon|saxophone|trumpet|trombone|tuba|french horn|
drums|organ|accordion|harmonica|banjo|mandolin|ukulele|sitar|xylophone|marimba|glockenspiel|timpani|
tambourine|bagpipes|recorder|piccolo|cornet|euphonium|harpsichord|lute|zither|synthesizer|bongos|cymbals|
triangle|didgeridoo|balalaika|bouzouki|tabla|theremin|ocarina|double bass|kazoo|castanets|dulcimer|sousaphone|
koto|erhu|oud|kalimba|djembe|melodica|bugle|celesta|concertina|steel drum|pan flute|shamisen|guzheng|pipa|
kora|veena|sarod|shehnai|bansuri|harmonium|santoor|mridangam|dhol|tanpura|sarangi|shakuhachi|biwa|taiko|
gayageum|haegeum|dizi|suona|sheng|yangqin|guqin|morin khuur|dombra|kemenche|saz|ney|qanun|darbuka|riq|bodhran|
uilleann pipes|tin whistle|hurdy gurdy|nyckelharpa|kantele|hardanger fiddle|alphorn|cimbalom|gusli|domra|
bayan|charango|cuatro|quena|cajon|guiro|maracas|claves|cowbell|conga|timbales|berimbau|cavaquinho|pandeiro|
balafon|talking drum|shekere|udu|ngoni|valiha|lyre|psaltery|sackbut|shawm|crumhorn|serpent|viola da gamba|
clavichord|spinet|pipe organ|electric guitar|bass guitar|flugelhorn|mellophone|bass clarinet|contrabassoon|
english horn|snare drum|vibraphone|tubular bells|handbells|gong|washboard|jaw harp|musical saw|slide whistle|
steel guitar|keytar|ondes martenot|mandola|banjolele|autoharp|calliope|glass harmonica|carillon|bandoneon|
fife|tabor|rebab|zurna|duduk|kaval|gaida|bombarde|rainstick|sleigh bells|wood block|alto saxophone|
tenor saxophone|baritone saxophone|soprano saxophone|bass drum|tenor drum|hand drum|frame drum|electric piano|
grand piano|upright piano|pump organ|chamber organ|twelve string guitar|classical guitar|slide guitar|
resonator guitar|bass trombone|piccolo trumpet|natural horn|wagner tuba|contrabass clarinet|basset horn|
alto flute|bass flute|octobass|kettle drum|bell tree|thumb piano|nose flute|bone flute|conch shell|log drum|
slit drum|water phone|
""")

SPORTS = _split("""
soccer|tennis|golf|basketball|baseball|cricket|rugby|hockey|volleyball|badminton|squash|table tennis|handball|
lacrosse|polo|water polo|netball|softball|darts|snooker|billiards|curling|croquet|dodgeball|ultimate frisbee|
field hockey|ice hockey|beach volleyball|racquetball|pickleball|football|hurling|kabaddi|bocce|shuffleboard|
futsal|korfball|floorball|bandy|rounders|paddle tennis|roller hockey|kickball|sepak takraw|jai alai|padel|
bowling|lawn bowls|skittles|cornhole|horseshoes|petanque|disc golf|minigolf|pool|carrom|foosball|air hockey|
gaelic football|australian rules football|american football|rugby league|canadian football|flag football|
wheelchair basketball|wheelchair rugby|sledge hockey|goalball|boccia|beach soccer|beach handball|
underwater hockey|underwater rugby|quadball|tchoukball|hornussen|pesapallo|gateball|chinlone|bossaball|
footvolley|teqball|spikeball|woodball|shinty|camogie|stickball|stoolball|inline hockey|ringette|broomball|
street hockey|cycle ball|bike polo|canoe polo|horseball|buzkashi|pato|pushball|fistball|speedball|tamburello|
real tennis|platform tennis|soft tennis|frontenis|fives|rackets|basque pelota|indoor cricket|wiffle ball|kubb|
molkky|quoits|skee ball|table football|mini football|five a side football|walking football|arena polo|
snow polo|touch rugby|tag rugby|beach rugby|beach tennis|touch football|indoor soccer|street soccer|
sitting volleyball|snow volleyball|power volleyball|underwater football|ball hockey|pond hockey|paddleball|
platform ball|keepie uppie|tennis polo|elephant polo|camel polo|yak polo|kin ball|
""")

# App. B: word set i has the two names TWO_FACT_NAMES[i] and the contents TWO_FACT_FOODS[i]; sets 0-7 calibrate.

TWO_FACT_NAMES = [
    ('Amy', 'Ben'), ('Clara', 'Daniel'), ('Ella', 'Felix'), ('Hannah', 'Isaac'), ('Jane', 'Kyle'), ('Laura', 'Mark'),
    ('Nina', 'Oscar'), ('Petra', 'Quinn'), ('Ruby', 'Simon'), ('Tessa', 'Umar'), ('Vera', 'Walter'), ('Yara', 'Zach'),
    ('Anna', 'Brian'), ('Chloe', 'Dylan'), ('Eva', 'Gavin'), ('Holly', 'Ian'), ('Jenny', 'Karl'), ('Lucy', 'Martin'),
    ('Nora', 'Owen'), ('Penny', 'Ryan'), ('Sarah', 'Tom'), ('Ursula', 'Vincent'), ('Willow', 'Xavier'), ('Yvonne', 'Zoe'),
    ('Aaron', 'Beth'), ('Colin', 'Daisy'), ('Edward', 'Freya'), ('Gordon', 'Heidi'), ('Irene', 'James'), ('Kelly', 'Liam'),
    ('Megan', 'Nathan'), ('Paige', 'Robert'), ('Sophie', 'Travis'), ('Vanessa', 'Wesley'), ('Alan', 'Brenda'), ('Carl', 'Donna'),
    ('Ethan', 'Faith'), ('Greg', 'Hugo'), ('Ingrid', 'Jason'), ('Karen', 'Lewis'), ('Monica', 'Neil'), ('Opal', 'Peter'),
    ('Rachel', 'Scott'), ('Tara', 'Wade'), ('Uma', 'Nick'), ('Abby', 'Bruce'), ('Cindy', 'Derek'), ('Elena', 'Fred'),
    ('Gina', 'Harold'), ('Isla', 'Joel'), ('Kara', 'Luke'), ('Maria', 'Norman'), ('Oliver', 'Pamela'), ('Roger', 'Sandra'),
    ('Troy', 'Violet'), ('Warren', 'Agnes'), ('Blake', 'Cora'), ('Dean', 'Edith'), ('Floyd', 'Gloria'), ('Harvey', 'Ada'),
    ('Jake', 'Lena'), ('Miles', 'Omar'), ('Phil', 'Rita'), ('Steve', 'Tanya'), ('Wayne', 'Alma'), ('Boris', 'Celia'),
    ('Dennis', 'Elsa'), ('Glen', 'Hilda'), ('Joan', 'Lars'), ('Maya', 'Nadia'), ('Priya', 'Raj'), ('Stella', 'Tim'),
]

TWO_FACT_FOODS = [
    ['broccoli', 'spinach', 'pasta', 'rice'], ['cookie', 'mango', 'potato', 'walnut'], ['cherry', 'kiwi', 'grape', 'apple'],
    ['peach', 'tomato', 'banana', 'cake'], ['cheese', 'butter', 'noodle', 'melon'], ['carrot', 'orange', 'muffin', 'plum'],
    ['almond', 'pear', 'bread', 'chicken'], ['onion', 'salmon', 'yogurt', 'lemon'], ['dumpling', 'risotto', 'tuna', 'guava'],
    ['steak', 'asparagus', 'custard', 'pomegranate'], ['persimmon', 'date', 'falafel', 'shallot'], ['pepper', 'corn', 'okra', 'fudge'],
    ['cracker', 'pea', 'rhubarb', 'lamb'], ['squash', 'burrito', 'lasagna', 'pancake'], ['lentil', 'sausage', 'burger', 'mushroom'],
    ['fennel', 'jam', 'scallop', 'crepe'], ['turkey', 'sushi', 'beet', 'cupcake'], ['tart', 'scallion', 'blueberry', 'olive'],
    ['chestnut', 'squid', 'ham', 'eggplant'], ['cantaloupe', 'pecan', 'quinoa', 'anchovy'], ['jelly', 'raspberry', 'celery', 'tofu'],
    ['fig', 'sorbet', 'tortilla', 'peanut'], ['caramel', 'ravioli', 'soup', 'grapefruit'], ['avocado', 'quince', 'curry', 'hummus'],
    ['hazelnut', 'blackberry', 'ginger', 'tangerine'], ['egg', 'prune', 'marshmallow', 'salad'], ['candy', 'artichoke', 'leek', 'trout'],
    ['duck', 'clam', 'raisin', 'pretzel'], ['cashew', 'toast', 'biscuit', 'pineapple'], ['turnip', 'popcorn', 'mussel', 'brownie'],
    ['bagel', 'apricot', 'crab', 'meatball'], ['waffle', 'strawberry', 'chocolate', 'zucchini'], ['baguette', 'cream', 'watermelon', 'porridge'],
    ['garlic', 'nectarine', 'lime', 'beef'], ['scone', 'cheesecake', 'lychee', 'pie'], ['cod', 'toffee', 'granola', 'parsnip'],
    ['honey', 'lettuce', 'shrimp', 'cabbage'], ['plantain', 'cereal', 'kale', 'yam'], ['spaghetti', 'chickpea', 'quiche', 'taco'],
    ['pumpkin', 'cranberry', 'sardine', 'halibut'], ['sandwich', 'coconut', 'bacon', 'pudding'], ['radish', 'croissant', 'pistachio', 'barley'],
    ['pork', 'cauliflower', 'lobster', 'oatmeal'], ['cucumber', 'bean', 'stew', 'papaya'], ['garlic', 'lobster', 'oyster', 'cucumber'],
    ['sausage', 'fudge', 'bean', 'pudding'], ['apricot', 'popcorn', 'barley', 'cream'], ['prune', 'grapefruit', 'radish', 'beet'],
    ['guava', 'burrito', 'squash', 'trout'], ['lettuce', 'honey', 'custard', 'marshmallow'], ['tangerine', 'pepper', 'lentil', 'clam'],
    ['steak', 'fig', 'kale', 'tortilla'], ['oyster', 'yam', 'falafel', 'lamb'], ['pumpkin', 'shrimp', 'chickpea', 'baguette'],
    ['asparagus', 'lychee', 'olive', 'mushroom'], ['cantaloupe', 'bacon', 'scallion', 'pomegranate'], ['persimmon', 'candy', 'scallop', 'quiche'],
    ['pancake', 'ginger', 'watermelon', 'corn'], ['egg', 'tart', 'squid', 'cranberry'], ['risotto', 'crepe', 'halibut', 'dumpling'],
    ['oatmeal', 'cauliflower', 'artichoke', 'scone'], ['plantain', 'stew', 'hazelnut', 'blackberry'], ['beef', 'pecan', 'cheesecake', 'sushi'],
    ['pretzel', 'bagel', 'meatball', 'toffee'], ['date', 'raspberry', 'strawberry', 'crab'], ['quinoa', 'eggplant', 'cracker', 'pistachio'],
    ['jam', 'quince', 'rhubarb', 'sorbet'], ['ravioli', 'jelly', 'nectarine', 'taco'], ['brownie', 'pea', 'cabbage', 'salad'],
    ['lime', 'zucchini', 'pork', 'cashew'], ['croissant', 'sardine', 'mussel', 'okra'], ['anchovy', 'burger', 'chocolate', 'pineapple'],
]

# App. G.3: sixteen verb frames ({N} = name, {F} = content); none uses 'eats'. Content-first frames: 5, 7, 9, 14.
PROSE_FRAMES = [
    '{N} ordered the {F}.',
    'For lunch, {N} had the {F}.',
    "{N}'s plate held the {F}.",
    '{N} finished the {F}.',
    'When the menu came around, {N} asked for the {F}.',
    'The {F} was what {N} picked.',
    '{N} went straight for the {F} without a second thought.',
    'It was the {F} that {N} wanted.',
    '{N} settled on the {F}.',
    'The {F} went to {N}.',
    'After some deliberation, {N} chose the {F}.',
    '{N} happily tucked into the {F}.',
    'Nobody was surprised when {N} got the {F}.',
    '{N} was served the {F}.',
    'The {F} ended up in front of {N}.',
    '{N} had a generous helping of the {F}.',
]

PROSE_FILLERS = [
    'The weather was mild.',
    'Outside, a bus rumbled past the window.',
    'A waiter hurried by with a stack of menus.',
    'Music played quietly from a speaker in the corner.',
    'It had rained earlier that morning, and the streets were still wet.',
    'The place was busier than usual.',
    'Someone at the next table laughed loudly.',
    'The conversation drifted toward plans for the weekend.',
    'A dog barked somewhere down the street.',
    'The tablecloths were freshly ironed.',
    'Sunlight came through the tall windows.',
    'Everyone agreed that the service was quick.',
    'A child near the door was drawing with crayons.',
    'The clock on the wall ran five minutes fast.',
    'Nobody was in a hurry that afternoon.',
    'A delivery van stopped outside for a few minutes and then drove off.',
    'The chairs were old but surprisingly comfortable.',
    'There was a short debate about a film they had all seen.',
    'The owner stopped by to say hello.',
    'A light breeze moved the curtains.',
    'Traffic was slow on the main road.',
    'Someone mentioned an upcoming birthday.',
    'The room smelled faintly of fresh paint.',
    'A phone buzzed on the table, but nobody answered it.',
]

PROSE_OPENINGS = [
    'The friends met for lunch at a small place on the corner.',
    'It was a busy Saturday at the market hall.',
    'The group had booked a table near the window.',
    'After a long walk, everyone was hungry.',
    'The new restaurant had only opened the week before.',
    'They had agreed to try somewhere different this time.',
    'The office outing started a little late.',
    'The evening was relaxed and unhurried.',
]

PROSE_CLOSINGS = [
    'Afterwards, they all walked to the station together.',
    'The bill was split evenly.',
    'It was dark by the time they left.',
    'They promised to come back soon.',
    'Everyone left in a good mood.',
    'The evening ended earlier than planned.',
]
