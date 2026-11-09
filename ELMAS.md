# ELMAS.md — Charte de développement ElmasCore

> Ce fichier est la signature d'un projet ElmasCore. Il définit les règles non négociables
> à respecter par tout contributeur à ce dépôt. Il doit être lu
> avant toute génération de code, toute refonte, tout ajout de dépendance.
>
> "From Data to Systems" — un projet ElmasCore se reconnaît à sa rigueur, pas à sa rapidité d'exécution.

---

## 1. Architecture

- L'architecture existante du projet fait autorité. On ne la contourne pas, on ne la
  duplique pas en parallèle, on ne crée pas un second pattern "plus simple" à côté de
  celui en place.
- Toute modification structurelle (nouveau dossier racine, changement de convention de
  nommage, nouvelle couche) doit être justifiée en une phrase avant d'être appliquée, pas
  après.
- Séparation stricte des responsabilités : UI, logique métier, accès aux données et
  configuration ne vivent jamais dans le même fichier. Un composant qui fait une requête
  réseau, calcule une règle métier et gère son propre style est un composant à corriger,
  pas un raccourci acceptable.
- Pas de dépendance ajoutée "parce que c'est plus rapide". Toute nouvelle librairie doit
  répondre à un besoin qu'aucune brique déjà présente ne couvre.

### Où va le code

- L'organisation est **par domaine métier**, pas par type technique. `students/`,
  `payments/`, `attendance/` plutôt que `components/`, `hooks/`, `utils/` en vrac à la
  racine. Le type technique (composant, service, hook) est un sous-dossier à l'intérieur
  du domaine, jamais l'inverse.
- Chaque nouveau fichier a un seul dossier "logique" possible avant d'être créé. Si
  l'endroit où le ranger est ambigu, c'est que le découpage du domaine doit être précisé
  d'abord — on ne le pose pas "quelque part" en attendant.
- Le flux de dépendances est à sens unique : UI → services/logique métier → accès aux
  données. Une couche basse (accès données) n'importe jamais une couche haute (UI). Aucune
  dépendance circulaire, jamais, même "temporairement".
- `utils/`, `helpers/`, `misc/`, `common/` sont interdits comme fourre-tout. Si une
  fonction n'a pas de domaine clair, c'est qu'elle est mal nommée ou mal pensée, pas
  qu'elle mérite un dossier générique.

## 2. Taille et complexité du code

- **350 lignes maximum par fichier**, sans exception de confort. Un fichier qui approche
  cette limite est un signal : il fait probablement plusieurs choses et doit être découpé
  avant d'être étendu, pas après.
- Une fonction, une responsabilité. Si une fonction a besoin d'un commentaire pour expliquer
  "d'abord on fait X, puis Y, puis Z", c'est qu'elle doit devenir trois fonctions.
- Pas de duplication silencieuse : si une logique existe déjà ailleurs dans le projet, on
  la réutilise ou on l'extrait, on ne la recopie pas.

## 3. Prévention du code spaghetti (Anti-Spaghetti & Clarté du flux)

- **Clauses de garde (Guard Clauses) et sortie précoce (Early Return)** :
  - **3 niveaux d'indentation maximum par fonction**, sans exception.
  - La "pyramide du destin" (`if` imbriqué dans un `if`, lui-même dans une boucle, elle-même dans un `switch`) est un rejet immédiat.
  - On traite d'abord les cas limites, les validations, les erreurs et les valeurs nulles en début de fonction avec un `return` ou `throw` immédiat. Le chemin nominal s'exécute toujours au niveau d'indentation 0 ou 1.
- **Pureté et effets de bord maîtrisés** :
  - Toute fonction de calcul, de validation, de filtrage ou de transformation de données **doit être pure** (déterministe, zéro mutation externe).
  - **Interdiction de muter silencieusement les arguments** passés en paramètre. Un paramètre entrant est considéré en lecture seule. Si une modification est nécessaire, on retourne une nouvelle structure immuable ou explicitement clonée.
  - **Zéro variable globale ou état mutable partagé sauvage**. Aucun singleton fourre-tout qui stocke des états modifiables de partout. L'état a un propriétaire unique et se transmet de manière explicite.
- **Signatures et abstraction homogène (Single Level of Abstraction - SLA)** :
  - **3 paramètres maximum par fonction**. Au-delà de 3, les arguments sont regroupés dans un objet ou une interface de configuration typée et validée.
  - Une fonction = un seul niveau d'abstraction.
  - **Pas de drapeaux de contrôle multiples (Boolean flags)**.
- **Couplage lâche et loi de Déméter** :
  - **Flux unidirectionnel strict** : aucune dépendance cyclique directe ou indirecte entre fichiers.

## 4. Choix technique

- On choisit **la meilleure stack pour le problème posé, jamais la plus rapide à sortir**.
- Avant d'écrire la première ligne, on nomme explicitement pourquoi cette stack et pas une
  autre.

## 5. UI / UX

- Densité et lisibilité priment sur la décoration.
- Chiffres, montants, identifiants : toujours en police à chiffres tabulaires, alignés
  correctement.

## 6. Nommage des fichiers et des fonctions

- **Fichiers** : le nom décrit ce que contient le fichier, jamais son type générique. Un nom de fichier générique (`helpers`, `misc`, `temp`, `final`, `v2`) est un rejet automatique.
- **Fonctions** : `verbe + nom`, qui décrit ce que la fonction fait, pas comment elle le fait.
  - Fonctions booléennes : préfixe `is`, `has`, `can`.
- Aucune abréviation qui n'est pas un standard du domaine.

## 7. Commentaires

- Un commentaire explique **pourquoi**, jamais **quoi**.
- Pas de code commenté laissé en place "au cas où".
- Commentaire de documentation obligatoire sur toute fonction exportée qui touche à une règle métier : ce qu'elle prend, ce qu'elle retourne, les cas limites gérés.

## 8. Qualité et maintenabilité générale

- Toute erreur est gérée explicitement, jamais avalée silencieusement.
- Pas de duplication de logique métier entre fichiers : une règle vit à un seul endroit.
- Tout code mort, inutilisé, obsolète ou expérimental est éliminé sans pitié.

## 9. Sécurité et données

- Aucune donnée sensible (clé API, identifiant, mot de passe) en dur dans le code ou
  versionnée. Toujours via variables d'environnement ou secret manager.

## 10. Documentation et livraison

- Un README à jour à la racine : ce que fait le projet, comment le lancer, comment
  contribuer. Un projet sans README à jour n'est pas considéré comme livrable.
- Les décisions structurantes sont consignées.

---

— **Elmas**, ElmasCore

