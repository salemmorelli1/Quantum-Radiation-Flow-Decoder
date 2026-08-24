# Upload with Git Bash

Create an empty GitHub repository named `Quantum-Radiation-Flow-Decoder`, then:

```bash
cd "/c/Users/salem/GitHub/Quantum Radiation Flow Decoder"

git init
git add .
git status
git commit -m "Initial quantum radiation flow research framework"
git branch -M main
git remote add origin https://github.com/salemmorelli1/Quantum-Radiation-Flow-Decoder.git
git push -u origin main
```

For GitHub Pages, choose **Settings -> Pages -> Deploy from a branch**, then
select branch `main` and folder `/docs`.

The site will be:

<https://salemmorelli1.github.io/Quantum-Radiation-Flow-Decoder/>
