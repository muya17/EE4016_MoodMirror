# Intro Script v2 (Casual / YOLO Style, Under 2 Minutes)

Hey everyone, I am Zach, and this is MoodMirror by Group 10.

Quick setup before we start: I will show one simple visual with the same person in three expressions. Top left, there is a speech bubble that says, "How is uni going?" Then the same person replies three different ways: happy, tired or sleepy, and confused.

The point is simple: even with the same words, facial expression changes the meaning a lot. You can say, "University is so fun. I love my 9AM classes," but the face tells the real story.

See? No one needed subtitles. Faces are basically high-speed human communication.

So that is the idea behind our project: can a practical AI system read facial expressions reliably without needing a giant, expensive model?

Because right now the trade-off is kind of painful:
- Big models are strong, but they are heavy and expensive.
- Tiny edge models are efficient, but accuracy can drop fast.

We asked a simple question: can we find a middle ground that is actually usable day to day?

Why does this matter? Real applications include road safety, fatigue or stress monitoring, human-computer interaction, and wellbeing support systems. Not to replace people, but to add an early warning layer when behavior looks concerning.

In this project, we built multiple FER pipelines, compared them properly, and integrated them into a live web app that you will see in a minute.

So today, we will take you through what we built, what worked, what did not, and then show the live demo.

If the demo behaves, we look smart. If not, we call it a stress test. Hopefully, you won't read our faces as stressed. 