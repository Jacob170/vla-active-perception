# LIBERO Classic Tasks and their LIBERO-PRO Perturbations

Categories 1-4 are the four suites LIBERO-PRO covers. Category 5 (LIBERO-90) is listed for completeness of the classic benchmark but **has no LIBERO-PRO perturbations** - it is absent from all five `libero_ood/*.yaml` files.

Perturbation source: `github.com/Zxy-MLlab/LIBERO-PRO/tree/master/libero_ood`

| File | Dimension | Config flag |
|---|---|---|
| `ood_object.yaml` | Object - swap an object asset for a variant (colour, size, shape) | `use_object` |
| `ood_spatial_relation.yaml` | Position - relocate an object to another object's spatial region | `use_swap` |
| `ood_language.yaml` | Semantic - paraphrase the instruction, task unchanged | `use_language` |
| `ood_task.yaml` | Task - redefine the goal predicate and target objects | `use_task` (cannot combine with others) |
| `ood_environment.yaml` | Environment - substitute the working surface / scene | `use_environment` |

---

## Category 1 - LIBERO-Goal (`libero_goal`)

Goal-conditioned tasks with changing targets. Episode cap: 300 steps.

### 1.1

**Prompt:** "open the middle drawer of the cabinet"

**bddl key:** `open_the_middle_drawer_of_the_cabinet`

**1.1 PRO**

- **Object** - replace:
  - `wooden_cabinet` -> `yellow_cabinet`, `white_cabinet`
- **Position** - relocate to another object's region:
  - `wooden_cabinet_1` -> `flat_stove_1`, `wine_rack_1`
- **Semantic** - paraphrases:
  - "open the cabinet’s middle drawer"
  - "pull the middle drawer of the cabinet"
  - "slide out the middle drawer"
- **Task** - redefined goal:
  - "open the bottom drawer of the cabinet"
    - goal: `(And (Open wooden_cabinet_1_bottom_region))`
    - obj_of_interest: `wooden_cabinet_1_bottom_region`
  - "open the top drawer of the cabinet"
    - goal: `(And (Open wooden_cabinet_1_top_region))`
    - obj_of_interest: `wooden_cabinet_1_top_region`
- **Environment** - substitute surface: `main_table`

### 1.2

**Prompt:** "put the cream cheese in the bowl"

**bddl key:** `put_the_cream_cheese_in_the_bowl`

**1.2 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`, `red_akita_black_bowl`, `red_bowl`, `white_bowl`, `yellow_bowl`
  - `cream_cheese` -> `red_cream_cheese`
- **Position** - relocate to another object's region:
  - `cream_cheese_1` -> `wine_bottle_1`, `plate_1`, `flat_stove_1`
  - `akita_black_bowl_1` -> `wine_bottle_1`, `plate_1`
- **Semantic** - paraphrases:
  - "place the cream cheese inside the bowl"
  - "put the cream cheese into the bowl"
  - "drop the cream cheese in the bowl"
- **Task** - redefined goal:
  - "put the wine bottle in the bowl"
    - goal: `(And (On wine_bottle_1 akita_black_bowl_1))`
    - obj_of_interest: `wine_bottle_1`, `akita_black_bowl_1`
  - "put the cream cheese on the top of the drawer"
    - goal: `(And (On cream_cheese_1 wooden_cabinet_1_top_side))`
    - obj_of_interest: `cream_cheese_1`, `wooden_cabinet_1_top_side`
- **Environment** - substitute surface: `main_table`

### 1.3

**Prompt:** "open the top drawer and put the bowl inside"

**bddl key:** `open_the_top_drawer_and_put_the_bowl_inside`

**1.3 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`, `red_akita_black_bowl`, `red_bowl`, `white_bowl`, `yellow_bowl`
  - `wooden_cabinet` -> `yellow_cabinet`, `white_cabinet`
- **Position** - relocate to another object's region:
  - `akita_black_bowl_1` -> `wine_bottle_1`, `plate_1`, `cream_cheese_1`
- **Semantic** - paraphrases:
  - "slide open the top drawer and place the bowl in it"
  - "put the bowl inside the top drawer after opening it"
  - "open the cabinet’s top drawer and insert the bowl"
- **Task** - redefined goal:
  - "Open the top layer of the drawer and put the cream cheese inside"
    - goal: `(And (In cream_cheese_1 wooden_cabinet_1_top_region))`
    - obj_of_interest: `cream_cheese_1`, `wooden_cabinet_1_top_region`
  - "Open the middle layer of the drawer and put the bowl inside"
    - goal: `(And (In akita_black_bowl_1 wooden_cabinet_1_middle_region))`
    - obj_of_interest: `akita_black_bowl_1`, `wooden_cabinet_1_middle_region`
- **Environment** - substitute surface: `main_table`

### 1.4

**Prompt:** "put the wine bottle on the rack"

**bddl key:** `put_the_wine_bottle_on_the_rack`

**1.4 PRO**

- **Object** - replace:
  - `wine_bottle` -> `white_bottle`
  - `wine_rack` -> `brown_rack`, `wine_rack_stand`
- **Position** - relocate to another object's region:
  - `wine_bottle_1` -> `akita_black_bowl_1`, `plate_1`, `cream_cheese_1`
  - `wine_rack_1` -> `wooden_cabinet_1`
- **Semantic** - paraphrases:
  - "place the wine bottle on the rack"
  - "put the wine bottle onto the rack"
  - "set the wine bottle down on the rack"
- **Task** - redefined goal:
  - "Put the cream cheese on the rack"
    - goal: `(And (On cream_cheese_1 wine_rack_1_top_region))`
    - obj_of_interest: `cream_cheese_1`, `wine_rack_1_top_region`
  - "Put the wine bottle on the stove"
    - goal: `(And (On wine_bottle_1 flat_stove_1_cook_region))`
    - obj_of_interest: `wine_bottle_1`, `flat_stove_1_cook_region`
- **Environment** - substitute surface: `main_table`

### 1.5

**Prompt:** "push the plate to the front of the stove"

**bddl key:** `push_the_plate_to_the_front_of_the_stove`

**1.5 PRO**

- **Object** - replace:
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `wine_bottle_1`, `akita_black_bowl_1`, `cream_cheese_1`
- **Semantic** - paraphrases:
  - "move the plate to the stove’s front"
  - "push the plate forward to the stove"
  - "slide the plate toward the front of the stove"
- **Task** - redefined goal:
  - "Push the cream cheese to the front of the stove"
    - goal: `(And (On cream_cheese_1 main_table_stove_front_region))`
    - obj_of_interest: `cream_cheese_1`, `main_table_stove_front_region`
- **Environment** - substitute surface: `main_table`

### 1.6

**Prompt:** "put the wine bottle on top of the cabinet"

**bddl key:** `put_the_wine_bottle_on_top_of_the_cabinet`

**1.6 PRO**

- **Object** - replace:
  - `wine_bottle` -> `white_bottle`
  - `wooden_cabinet` -> `yellow_cabinet`, `white_cabinet`
- **Position** - relocate to another object's region:
  - `wine_bottle_1` -> `akita_black_bowl_1`, `plate_1`, `cream_cheese_1`
  - `wooden_cabinet_1` -> `flat_stove_1`, `wine_rack_1`
- **Semantic** - paraphrases:
  - "place the wine bottle on top of the cabinet"
  - "put the wine bottle above the cabinet"
  - "set the wine bottle on the cabinet top"
- **Task** - redefined goal:
  - "put the wine bottle in the bowl"
    - goal: `(And (On wine_bottle_1 akita_black_bowl_1))`
    - obj_of_interest: `wine_bottle_1`, `akita_black_bowl_1`
- **Environment** - substitute surface: `main_table`

### 1.7

**Prompt:** "put the bowl on the plate"

**bddl key:** `put_the_bowl_on_the_plate`

**1.7 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`, `red_akita_black_bowl`, `red_bowl`, `white_bowl`, `yellow_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `akita_black_bowl_1` -> `wine_bottle_1`, `cream_cheese_1`
  - `plate_1` -> `wine_bottle_1`, `cream_cheese_1`
- **Semantic** - paraphrases:
  - "place the bowl on top of the plate"
  - "put the bowl onto the plate"
  - "set the bowl down on the plate"
- **Task** - redefined goal:
  - "Put the wine bottle on the plate"
    - goal: `(And (On wine_bottle_1 plate_1))`
    - obj_of_interest: `wine_bottle_1`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 1.8

**Prompt:** "put the bowl on the stove"

**bddl key:** `put_the_bowl_on_the_stove`

**1.8 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`, `red_akita_black_bowl`, `red_bowl`, `white_bowl`, `yellow_bowl`
  - `flat_stove` -> `yellow_stove`
- **Position** - relocate to another object's region:
  - `akita_black_bowl_1` -> `wine_bottle_1`, `plate_1`, `cream_cheese_1`
  - `flat_stove_1` -> `cream_cheese_1`, `wooden_cabinet_1`
- **Semantic** - paraphrases:
  - "place the bowl on the stove"
  - "put the bowl onto the stove"
  - "set the bowl down on the stove"
- **Task** - redefined goal:
  - "Put the plate on the stove"
    - goal: `(And (On plate_1 flat_stove_1_cook_region))`
    - obj_of_interest: `plate_1`, `flat_stove_1_cook_region`
- **Environment** - substitute surface: `main_table`

### 1.9

**Prompt:** "turn on the stove"

**bddl key:** `turn_on_the_stove`

**1.9 PRO**

- **Object** - replace:
  - `flat_stove` -> `yellow_stove`
- **Position** - relocate to another object's region:
  - `flat_stove_1` -> `wooden_cabinet_1`, `cream_cheese_1`
- **Semantic** - paraphrases:
  - "switch on the stove"
  - "turn the stove on"
  - "ignite the stove"
- **Task** - redefined goal:
  - "Turn off the stove"
    - goal: `(And (Turnoff flat_stove_1))`
    - obj_of_interest: `flat_stove_1`
- **Environment** - substitute surface: `main_table`

### 1.10

**Prompt:** "put the bowl on top of the cabinet"

**bddl key:** `put_the_bowl_on_top_of_the_cabinet`

**1.10 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`, `red_akita_black_bowl`, `red_bowl`, `white_bowl`, `yellow_bowl`
  - `wooden_cabinet` -> `yellow_cabinet`, `white_cabinet`
- **Position** - relocate to another object's region:
  - `akita_black_bowl_1` -> `wine_bottle_1`, `plate_1`, `cream_cheese_1`
  - `wooden_cabinet_1` -> `flat_stove_1`, `wine_rack_1`
- **Semantic** - paraphrases:
  - "place the bowl on top of the cabinet"
  - "put the bowl above the cabinet"
  - "set the bowl down on the cabinet top"
- **Task** - redefined goal:
  - "Put the plate on the top of the drawer"
    - goal: `(And (On plate_1 wooden_cabinet_1_top_side))`
    - obj_of_interest: `plate_1`, `wooden_cabinet_1_top_side`
- **Environment** - substitute surface: `main_table`

---

## Category 2 - LIBERO-Spatial (`libero_spatial`)

Tasks requiring reasoning about spatial relations. Episode cap: 220 steps.

### 2.1

**Prompt:** "pick up the black bowl between the plate and the ramekin and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`

**2.1 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `cookies_1`, `akita_black_bowl_2`
- **Semantic** - paraphrases:
  - "lift the black bowl between the plate and ramekin and set it on the plate"
  - "pick the black bowl between plate and ramekin and put it on plate"
  - "move the black bowl from between plate and ramekin to the plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl not between the plate and the ramekin and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`
- **Environment** - substitute surface: `main_table`

### 2.2

**Prompt:** "pick up the black bowl from table center and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`

**2.2 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`
- **Semantic** - paraphrases:
  - "lift the black bowl from the table center and put it on the plate"
  - "move the black bowl at the center of table to the plate"
  - "pick up the black bowl in the middle of the table and place it on plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl next to the plate and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.3

**Prompt:** "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`

**2.3 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`
- **Semantic** - paraphrases:
  - "take the black bowl from the top drawer and put it on plate"
  - "lift the black bowl in the cabinet top drawer and place it on plate"
  - "pick up black bowl inside the top drawer and move it to plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl on the top of the wooden cabinet and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.4

**Prompt:** "pick up the black bowl next to the cookie box and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`

**2.4 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `glazed_rim_porcelain_ramekin_1`
- **Semantic** - paraphrases:
  - "grab the black bowl beside the cookie box and set it on plate"
  - "move black bowl near cookie box to the plate"
  - "pick the bowl next to cookie box and put on plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl on the stove and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.5

**Prompt:** "pick up the black bowl next to the plate and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`

**2.5 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `cookies_1`
- **Semantic** - paraphrases:
  - "pick the bowl beside the plate and put it on plate"
  - "grab the black bowl next to plate and move it to plate"
  - "lift bowl by plate and set it on the plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl next to the ramekin and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.6

**Prompt:** "pick up the black bowl next to the ramekin and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`

**2.6 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`
- **Semantic** - paraphrases:
  - "take black bowl next to ramekin and place on plate"
  - "move bowl beside ramekin to the plate"
  - "grab black bowl near ramekin and set it on plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl next to the cookie box and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.7

**Prompt:** "pick up the black bowl on the cookie box and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`

**2.7 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `wooden_cabinet_1`
- **Semantic** - paraphrases:
  - "lift black bowl from cookie box and put it on plate"
  - "pick bowl resting on cookie box and place it on plate"
  - "move black bowl off cookie box onto the plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl on the top of the cabinet and place it on the plate"
    - goal: `(On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.8

**Prompt:** "pick up the black bowl on the ramekin and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`

**2.8 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`
- **Semantic** - paraphrases:
  - "grab black bowl on ramekin and set it on plate"
  - "lift bowl from ramekin and move to plate"
  - "pick black bowl off ramekin and put onto plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl on the cookie box and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.9

**Prompt:** "pick up the black bowl on the stove and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`

**2.9 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`
- **Semantic** - paraphrases:
  - "lift bowl from stove and set on plate"
  - "move black bowl on stove onto plate"
  - "pick black bowl off stove and put on plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl on the top of the cabinet and place it on the plate"
    - goal: `(On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

### 2.10

**Prompt:** "pick up the black bowl on the wooden cabinet and place it on the plate"

**bddl key:** `pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`

**2.10 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `plate` -> `yellow_plate`
- **Position** - relocate to another object's region:
  - `plate_1` -> `akita_black_bowl_2`, `cookies_1`, `glazed_rim_porcelain_ramekin_1`
- **Semantic** - paraphrases:
  - "lift bowl from cabinet and put on plate"
  - "move black bowl sitting on cabinet onto plate"
  - "grab bowl on cabinet top and place on plate"
- **Task** - redefined goal:
  - "Pick the akita black bowl on the stove and place it on the plate"
    - goal: `(And (On akita_black_bowl_2 plate_1))`
    - obj_of_interest: `akita_black_bowl_2`, `plate_1`
- **Environment** - substitute surface: `main_table`

---

## Category 3 - LIBERO-Object (`libero_object`)

Tasks centered on manipulating different objects. Episode cap: 280 steps.

### 3.1

**Prompt:** "pick up the alphabet soup and place it in the basket"

**bddl key:** `pick_up_the_alphabet_soup_and_place_it_in_the_basket`

**3.1 PRO**

- **Object** - replace:
  - `alphabet_soup` -> `bigger_alphabet_soup`, `red_alphabet_soup`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `alphabet_soup_1` -> `salad_dressing_1`, `cream_cheese_1`, `milk_1`, `tomato_sauce_1`, `butter_1`
- **Semantic** - paraphrases:
  - "grab alphabet soup and put it into basket"
  - "place alphabet soup in basket"
  - "pick alphabet soup and drop into basket"
- **Task** - redefined goal:
  - "Pick the cream cheese and place it in the basket"
    - goal: `(And (In cream_cheese_1 basket_1_contain_region))`
    - obj_of_interest: `cream_cheese_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.2

**Prompt:** "pick up the bbq sauce and place it in the basket"

**bddl key:** `pick_up_the_bbq_sauce_and_place_it_in_the_basket`

**3.2 PRO**

- **Object** - replace:
  - `bbq_sauce` -> `green_bbq_sauce`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `bbq_sauce_1` -> `chocolate_pudding_1`, `ketchup_1`, `salad_dressing_1`, `alphabet_soup_1`, `cream_cheese_1`
- **Semantic** - paraphrases:
  - "pick up bbq sauce and put it into basket"
  - "place bbq sauce in basket"
  - "move bbq sauce into basket"
- **Task** - redefined goal:
  - "Pick the ketchup and place it in the basket"
    - goal: `(And (In ketchup_1 basket_1_contain_region))`
    - obj_of_interest: `ketchup_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.3

**Prompt:** "pick up the butter and place it in the basket"

**bddl key:** `pick_up_the_butter_and_place_it_in_the_basket`

**3.3 PRO**

- **Object** - replace:
  - `butter` -> `green_butter`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `butter_1` -> `tomato_sauce_1`, `orange_juice_1`, `chocolate_pudding_1`, `bbq_sauce_1`, `ketchup_1`
- **Semantic** - paraphrases:
  - "grab butter and put it into basket"
  - "place butter in basket"
  - "pick butter and drop into basket"
- **Task** - redefined goal:
  - "Pick the orange juice and place it in the basket"
    - goal: `(And (In orange_juice_1 basket_1_contain_region))`
    - obj_of_interest: `orange_juice_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.4

**Prompt:** "pick up the chocolate pudding and place it in the basket"

**bddl key:** `pick_up_the_chocolate_pudding_and_place_it_in_the_basket`

**3.4 PRO**

- **Object** - replace:
  - `chocolate_pudding` -> `green_chocolate_pudding`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `chocolate_pudding_1` -> `orange_juice_1`, `bbq_sauce_1`, `ketchup_1`, `salad_dressing_1`, `alphabet_soup_1`
- **Semantic** - paraphrases:
  - "lift chocolate pudding and put it in basket"
  - "place pudding in basket"
  - "move chocolate pudding into basket"
- **Task** - redefined goal:
  - "Pick the salad dressing and place it in the basket"
    - goal: `(And (In salad_dressing_1 basket_1_contain_region))`
    - obj_of_interest: `salad_dressing_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.5

**Prompt:** "pick up the cream cheese and place it in the basket"

**bddl key:** `pick_up_the_cream_cheese_and_place_it_in_the_basket`

**3.5 PRO**

- **Object** - replace:
  - `cream_cheese` -> `red_cream_cheese`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `cream_cheese_1` -> `alphabet_soup_1`, `milk_1`, `tomato_sauce_1`, `butter_1`, `orange_juice_1`
- **Semantic** - paraphrases:
  - "grab cream cheese and put into basket"
  - "place cream cheese in basket"
  - "drop cream cheese in basket"
- **Task** - redefined goal:
  - "Pick the alphabet soup and place it in the basket"
    - goal: `(And (In alphabet_soup_1 basket_1_contain_region))`
    - obj_of_interest: `alphabet_soup_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.6

**Prompt:** "pick up the ketchup and place it in the basket"

**bddl key:** `pick_up_the_ketchup_and_place_it_in_the_basket`

**3.6 PRO**

- **Object** - replace:
  - `ketchup` -> `blue_ketchup`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `ketchup_1` -> `bbq_sauce_1`, `salad_dressing_1`, `alphabet_soup_1`, `cream_cheese_1`, `milk_1`
- **Semantic** - paraphrases:
  - "lift ketchup and put it in basket"
  - "place ketchup in basket"
  - "move ketchup into basket"
- **Task** - redefined goal:
  - "Pick the milk and place it in the basket"
    - goal: `(And (In milk_1 basket_1_contain_region))`
    - obj_of_interest: `milk_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.7

**Prompt:** "pick up the milk and place it in the basket"

**bddl key:** `pick_up_the_milk_and_place_it_in_the_basket`

**3.7 PRO**

- **Object** - replace:
  - `milk` -> `yellow_milk`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `milk_1` -> `cream_cheese_1`, `tomato_sauce_1`, `butter_1`, `orange_juice_1`, `chocolate_pudding_1`
- **Semantic** - paraphrases:
  - "grab milk and put into basket"
  - "place milk in basket"
  - "drop milk into basket"
- **Task** - redefined goal:
  - "Pick the butter and place it in the basket"
    - goal: `(And (In butter_1 basket_1_contain_region))`
    - obj_of_interest: `butter_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.8

**Prompt:** "pick up the orange juice and place it in the basket"

**bddl key:** `pick_up_the_orange_juice_and_place_it_in_the_basket`

**3.8 PRO**

- **Object** - replace:
  - `milk` -> `bigger_milk`, `yellow_milk`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `orange_juice_1` -> `butter_1`, `chocolate_pudding_1`, `bbq_sauce_1`, `ketchup_1`, `salad_dressing_1`
- **Semantic** - paraphrases:
  - "pick orange juice and put into basket"
  - "place orange juice in basket"
  - "move orange juice into basket"
- **Task** - redefined goal:
  - "Pick the chocolate pudding and place it in the basket"
    - goal: `(And (In chocolate_pudding_1 basket_1_contain_region))`
    - obj_of_interest: `chocolate_pudding_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.9

**Prompt:** "pick up the salad dressing and place it in the basket"

**bddl key:** `pick_up_the_salad_dressing_and_place_it_in_the_basket`

**3.9 PRO**

- **Object** - replace:
  - `salad_dressing` -> `red_salad_dressing`, `new_salad_dressing`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `salad_dressing_1` -> `ketchup_1`, `alphabet_soup_1`, `cream_cheese_1`, `milk_1`, `tomato_sauce_1`
- **Semantic** - paraphrases:
  - "grab salad dressing and put into basket"
  - "place salad dressing in basket"
  - "move salad dressing into basket"
- **Task** - redefined goal:
  - "Pick the tomato sauce and place it in the basket"
    - goal: `(And (In tomato_sauce_1 basket_1_contain_region))`
    - obj_of_interest: `tomato_sauce_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

### 3.10

**Prompt:** "pick up the tomato sauce and place it in the basket"

**bddl key:** `pick_up_the_tomato_sauce_and_place_it_in_the_basket`

**3.10 PRO**

- **Object** - replace:
  - `tomato_sauce` -> `yellow_tomato_sauce`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `tomato_sauce_1` -> `milk_1`, `butter_1`, `orange_juice_1`, `chocolate_pudding_1`, `bbq_sauce_1`
- **Semantic** - paraphrases:
  - "pick tomato sauce and put it in basket"
  - "place tomato sauce into basket"
  - "move tomato sauce into basket"
- **Task** - redefined goal:
  - "Pick the bbq sauce and place it in the basket"
    - goal: `(And (In bbq_sauce_1 basket_1_contain_region))`
    - obj_of_interest: `bbq_sauce_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `floor`

---

## Category 4 - LIBERO-10 / LIBERO-Long (`libero_10`)

Long-horizon tasks from the LIBERO-100 collection. Episode cap: 520 steps.

### 4.1

**Prompt:** "turn on the stove and put the moka pot on it"

**bddl key:** `KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`  ·  **scene:** `KITCHEN_SCENE3`

**4.1 PRO**

- **Object** - replace:
  - `moka_pot` -> `yellow_moka_pot`
  - `flat_stove` -> `yellow_stove`
- **Position** - relocate to another object's region:
  - `moka_pot_1` -> `chefmate_8_frypan_1`
  - `flat_stove_1` -> `chefmate_8_frypan_1`
- **Semantic** - paraphrases:
  - "switch stove on and put moka pot on it"
  - "turn stove on and place moka pot"
  - "ignite stove and set moka pot there"
- **Task** - redefined goal:
  - "turn on the stove and put the pan on it"
    - goal: `And (Turnon flat_stove_1) (On chefmate_8_frypan_1 flat_stove_1_cook_region))`
    - obj_of_interest: `flat_stove_1`, `chefmate_8_frypan_1`, `flat_stove_1_cook_region`
- **Environment** - substitute surface: `kitchen_table`

### 4.2

**Prompt:** "put the black bowl in the bottom drawer of the cabinet and close it"

**bddl key:** `KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`  ·  **scene:** `KITCHEN_SCENE4`

**4.2 PRO**

- **Object** - replace:
  - `akita_black_bowl` -> `black_bowl`, `bigger_akita_black_bowl`
  - `white_cabinet` -> `wooden_cabinet`, `yellow_cabinet`, `short_cabinet`
- **Position** - relocate to another object's region:
  - `akita_black_bowl_1` -> `wine_bottle_1`, `wine_rack_1`
  - `white_cabinet_1` -> `wine_rack_1`
- **Semantic** - paraphrases:
  - "put black bowl inside bottom drawer and close it"
  - "place black bowl in cabinet’s bottom drawer and shut"
  - "insert black bowl in lower drawer and close"
- **Task** - redefined goal:
  - "put the bottle in the bottom drawer of the cabinet and close it"
    - goal: `(And (Close white_cabinet_1_bottom_region) (In wine_bottle_1 white_cabinet_1_bottom_region))`
    - obj_of_interest: `white_cabinet_1_bottom_region`, `wine_bottle_1`, `white_cabinet_1_bottom_region`
- **Environment** - substitute surface: `kitchen_table`

### 4.3

**Prompt:** "put the yellow and white mug in the microwave and close it"

**bddl key:** `KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`  ·  **scene:** `KITCHEN_SCENE6`

**4.3 PRO**

- **Object** - replace:
  - `white_yellow_mug` -> `white_porcelain_mug`
- **Position** - relocate to another object's region:
  - `white_yellow_mug_1` -> `porcelain_mug_1`
  - `microwave_1` -> `porcelain_mug_1`
- **Semantic** - paraphrases:
  - "place yellow and white mug in microwave and close door"
  - "put mug in microwave then shut it"
  - "insert yellow-white mug into microwave and close"
- **Task** - redefined goal:
  - "put the white mug in the microwave and close it"
    - goal: `(And (In porcelain_mug_1 microwave_1_heating_region) (Close microwave_1))`
    - obj_of_interest: `porcelain_mug_1`, `microwave_1_heating_region`, `microwave_1`
- **Environment** - substitute surface: `kitchen_table`

### 4.4

**Prompt:** "put both moka pots on the stove"

**bddl key:** `KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`  ·  **scene:** `KITCHEN_SCENE8`

**4.4 PRO**

- **Object** - replace:
  - `moka_pot` -> `yellow_moka_pot`
- **Position** - relocate to another object's region:
  - `moka_pot_1` -> `flat_stove_1`
  - `moka_pot_2` -> `flat_stove_1`
- **Semantic** - paraphrases:
  - "place both moka pots on stove"
  - "put moka pots onto stove"
  - "set two moka pots on top of stove"
- **Task** - redefined goal:
  - "put the left moka pot on the stove"
    - goal: `(And (On moka_pot_1 flat_stove_1_cook_region) (Turnon flat_stove_1))`
    - obj_of_interest: `moka_pot_1`, `flat_stove_1_cook_region`, `flat_stove_1`
  - "put the right moka pot on the stove"
    - goal: `(And (On moka_pot_2 flat_stove_1_cook_region) (Turnon flat_stove_1))`
    - obj_of_interest: `moka_pot_2`, `flat_stove_1_cook_region`, `flat_stove_1`
- **Environment** - substitute surface: `kitchen_table`

### 4.5

**Prompt:** "put both the alphabet soup and the cream cheese box in the basket"

**bddl key:** `LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`  ·  **scene:** `LIVING_ROOM_SCENE1`

**4.5 PRO**

- **Object** - replace:
  - `alphabet_soup` -> `bigger_alphabet_soup`, `red_alphabet_soup`
  - `cream_cheese` -> `red_cream_cheese`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `alphabet_soup_1` -> `tomato_sauce_1`, `ketchup_1`
  - `cream_cheese_1` -> `tomato_sauce_1`, `ketchup_1`
- **Semantic** - paraphrases:
  - "put alphabet soup and cream cheese box in basket"
  - "place soup and cream cheese box into basket"
  - "add cream cheese box and alphabet soup into basket"
- **Task** - redefined goal:
  - "put both the ketchup and the cream cheese box in the basket"
    - goal: `(And (In ketchup_1 basket_1_contain_region) (In cream_cheese_1 basket_1_contain_region))`
    - obj_of_interest: `ketchup_1`, `basket_1_contain_region`, `cream_cheese_1`
  - "put both the alphabet soup and the ketchup in the basket"
    - goal: `(And (In alphabet_soup_1 basket_1_contain_region) (In ketchup_1 basket_1_contain_region))`
    - obj_of_interest: `alphabet_soup_1`, `basket_1_contain_region`, `ketchup_1`
- **Environment** - substitute surface: `living_room_table`

### 4.6

**Prompt:** "put both the alphabet soup and the tomato sauce in the basket"

**bddl key:** `LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`  ·  **scene:** `LIVING_ROOM_SCENE2`

**4.6 PRO**

- **Object** - replace:
  - `alphabet_soup` -> `bigger_alphabet_soup`, `red_alphabet_soup`
  - `tomato_sauce` -> `yellow_tomato_sauce`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `alphabet_soup_1` -> `cream_cheese_1`, `ketchup_1`, `milk_1`, `orange_juice_1`, `butter_1`
  - `tomato_sauce_1` -> `cream_cheese_1`, `ketchup_1`, `milk_1`, `orange_juice_1`, `butter_1`
- **Semantic** - paraphrases:
  - "place alphabet soup and tomato sauce into basket"
  - "put tomato sauce and alphabet soup into basket"
  - "drop alphabet soup and tomato sauce inside basket"
- **Task** - redefined goal:
  - "put both the cream cheese and the tomato sauce in the basket"
    - goal: `(And (In cream_cheese_1 basket_1_contain_region) (In tomato_sauce_1 basket_1_contain_region))`
    - obj_of_interest: `cream_cheese_1`, `basket_1_contain_region`, `tomato_sauce_1`
  - "put both the alphabet soup and the cream cheese in the basket"
    - goal: `(And (In alphabet_soup_1 basket_1_contain_region) (In cream_cheese_1 basket_1_contain_region))`
    - obj_of_interest: `alphabet_soup_1`, `basket_1_contain_region`, `ream_cheese_1`
- **Environment** - substitute surface: `living_room_table`

### 4.7

**Prompt:** "put both the cream cheese box and the butter in the basket"

**bddl key:** `LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`  ·  **scene:** `LIVING_ROOM_SCENE2`

**4.7 PRO**

- **Object** - replace:
  - `cream_cheese` -> `red_cream_cheese`
  - `butter` -> `green_butter`
  - `basket` -> `red_basket`
- **Position** - relocate to another object's region:
  - `cream_cheese_1` -> `alphabet_soup_1`, `tomato_sauce_1`, `ketchup_1`, `milk_1`, `orange_juice_1`
  - `butter_1` -> `alphabet_soup_1`, `tomato_sauce_1`, `ketchup_1`, `milk_1`, `orange_juice_1`
- **Semantic** - paraphrases:
  - "place cream cheese box and butter into basket"
  - "put butter and cream cheese box in basket"
  - "add cream cheese box with butter into basket"
- **Task** - redefined goal:
  - "put both the alphabet soup and the butter in the basket"
    - goal: `(And (In alphabet_soup_1 basket_1_contain_region) (In butter_1 basket_1_contain_region))`
    - obj_of_interest: `alphabet_soup_1`, `butter_1`, `basket_1_contain_region`
  - "put both the cream cheese box and the alphabet soup in the basket"
    - goal: `(And (In cream_cheese_1 basket_1_contain_region) (In alphabet_soup_1 basket_1_contain_region))`
    - obj_of_interest: `cream_cheese_1`, `butter_1`, `basket_1_contain_region`
- **Environment** - substitute surface: `living_room_table`

### 4.8

**Prompt:** "put the white mug on the left plate and put the yellow and white mug on the right plate"

**bddl key:** `LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`  ·  **scene:** `LIVING_ROOM_SCENE5`

**4.8 PRO**

- **Object** - replace:
  - `plate` -> `yellow_plate`
  - `white_yellow_mug` -> `white_porcelain_mug`
  - `porcelain_mug` -> `white_porcelain_mug`
- **Position** - relocate to another object's region:
  - `white_yellow_mug_1` -> `red_coffee_mug_1`
  - `porcelain_mug_1` -> `red_coffee_mug_1`, `white_yellow_mug_1`
- **Semantic** - paraphrases:
  - "place white mug on left plate and yellow-white mug on right plate"
  - "put white mug on left plate and yellow-white mug on right one"
  - "{'set mugs': 'white on left plate, yellow-white on right plate'}"
- **Task** - redefined goal:
  - "put the yellow and white mug on the left plate and put the white mug on the right plate"
    - goal: `(And (On white_yellow_mug_1 plate_1) (On porcelain_mug_1 plate_2))`
    - obj_of_interest: `white_yellow_mug_1`, `porcelain_mug_1`, `plate_1`, `plate_2`
  - "put the white mug on the right plate and put the yellow and white mug on the left plate"
    - goal: `(And (On porcelain_mug_1 plate_2) (On white_yellow_mug_1 plate_1))`
    - obj_of_interest: `white_yellow_mug_1`, `porcelain_mug_1`, `plate_1`, `plate_2`
- **Environment** - substitute surface: `living_room_table`

### 4.9

**Prompt:** "put the white mug on the plate and put the chocolate pudding to the right of the plate"

**bddl key:** `LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`  ·  **scene:** `LIVING_ROOM_SCENE6`

**4.9 PRO**

- **Object** - replace:
  - `plate` -> `yellow_plate`
  - `chocolate_pudding` -> `green_chocolate_pudding`
  - `porcelain_mug` -> `white_porcelain_mug`
- **Position** - relocate to another object's region:
  - `chocolate_pudding_1` -> `red_coffee_mug_1`
  - `porcelain_mug_1` -> `red_coffee_mug_1`, `chocolate_pudding_1`
- **Semantic** - paraphrases:
  - "place white mug on plate and chocolate pudding to right"
  - "put chocolate pudding right of plate and white mug on plate"
  - "move mug onto plate and pudding to its right"
- **Task** - redefined goal:
  - "put the red mug on the plate and put the chocolate pudding to the right of the plate"
    - goal: `(And (On red_coffee_mug_1 plate_1) (On chocolate_pudding_1 living_room_table_plate_right_region))`
    - obj_of_interest: `red_coffee_mug_1`, `chocolate_pudding_1`, `plate_1`, `living_room_table_plate_right_region`
  - "put the white mug on the plate and put the chocolate pudding to the left of the plate"
    - goal: `(And (On porcelain_mug_1 plate_1) (On chocolate_pudding_1 living_room_table_plate_left_region))`
    - obj_of_interest: `porcelain_mug_1`, `chocolate_pudding_1`, `plate_1`, `living_room_table_plate_left_region`
- **Environment** - substitute surface: `living_room_table`

### 4.10

**Prompt:** "pick up the book and place it in the back compartment of the caddy"

**bddl key:** `STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`  ·  **scene:** `STUDY_SCENE1`

**4.10 PRO**

- **Object** - replace:
  - `black_book` -> `black_bowl`, `bigger_akita_black_bowl`
- **Position** - relocate to another object's region:
  - `black_book_1` -> `white_yellow_mug_1`
- **Semantic** - paraphrases:
  - "pick up the book and place in back compartment"
  - "put book in rear compartment of caddy"
  - "set book into back slot of caddy"
- **Task** - redefined goal:
  - "pick up the cup and place it in the back compartment of the caddy"
    - goal: `(And (In white_yellow_mug_1 desk_caddy_1_back_contain_region))`
    - obj_of_interest: `white_yellow_mug_1`, `desk_caddy_1_back_contain_region`
  - "pick up the book and place it in the front compartment of the caddy"
    - goal: `(And (In black_book_1 desk_caddy_1_front_contain_region))`
    - obj_of_interest: `black_book_1`, `desk_caddy_1_front_contain_region`
- **Environment** - substitute surface: `study_table`

---

## Category 5 - LIBERO-90 (`libero_90`)

Short-horizon tasks from the LIBERO-100 collection. Episode cap: 400 steps.

**No LIBERO-PRO perturbations exist for this suite.** 90 tasks, none covered by `libero_ood/*.yaml`.

---
