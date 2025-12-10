# game_of_social_graphs_and_thrones
Final project about Game of Thrones for the DTU course Social Graphs and Interactions (02805).

The full code for the project is assembled in ```final_code.py```, from where the entire analysis can also be run. 

To run the code, the necessary libraries need to be installed. An environment with the needed libraries can be created with ```conda env create -f environment.yaml```.

## Central Idea:
### Idea
Our main idea is analyzing the many connections in the *Game of Thrones* universe. We use k-clique community detection and analyze centrality by looking at weighted degrees of character nodes. We also do sentiment analysis on episode synopses, connecting the calculated sentiment scores to character connections.

### Why is it interesting?
Game of Thrones and House of the Dragon feature a vast universe, rich in characters, cast, episodes, books, lore, etc. The universe was really popularized by the HBO series Game of Thrones, which was adopted by the original book series *A Song of Ice and Fire* by George R. R. Martin. Using different network science tools and sentiment analysis, we uncover important characters, and the evolution of their relationships, which aligns with different story points.

### Dataset
To do our project we downloaded the pages that make up the [Wiki of Westeros](https://gameofthrones.fandom.com/wiki/Westeros).

The full network has:
* 5672 nodes
* 198768 edges
* Average degree of 35.04
* 847 character pages
* 91 episode pages from Game of Thrones and House of the Dragon

### Download process
To download the needed pages, we used the skills we've learned in the course about APIs.

