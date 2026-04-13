# Intro Script v2 (Casual / YOLO Style, Under 2 Minutes)

Hey everyone, I am Zach, and this is MoodMirror by Group 10.

Quick setup before we start: I will show 3 images with the same text reply, like "I am good". One image has a smile, one looks sad, one looks confused. Same words, completely different meaning.

See? No one needed subtitles. Faces are basically high-speed human communication.

So that is the idea behind our project: can a practical AI system read facial expressions reliably without needing a giant, expensive model?

Because right now the trade-off is kind of painful:
- Big models are strong, but they are heavy and expensive.
- Tiny edge models are efficient, but accuracy can drop fast.

We asked a simple question: can we find a middle ground that is actually usable day to day?

Why does this matter? Real applications include road safety, fatigue or stress monitoring, human-computer interaction, and wellbeing support systems. Not to replace people, but to add an early warning layer when behavior looks concerning.

In this project, we built multiple FER pipelines, compared them properly, and integrated them into a live web app that you will see in a minute.

So today, we will take you through what we built, what worked, what did not, and then show the live demo.

If the demo behaves, we look smart. If not, we call it a stress test.