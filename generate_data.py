
import csv
import random

import config

random.seed(42)

CATEGORIES = {
    "Access Issue": [
        ("Cannot log into my account",
         "I keep getting an invalid credentials error even after resetting my password."),
        ("Locked out after password reset",
         "I reset my password an hour ago but the system still says my account is locked."),
        ("Two factor authentication not working",
         "The OTP code never arrives on my registered phone number, so I can't sign in."),
        ("SSO login redirects to blank page",
         "When I click login with company SSO it redirects me to a blank white page."),
        ("New employee cannot access dashboard",
         "Our new hire was given credentials but the dashboard shows a permission denied screen."),
        ("Login page is extremely slow to respond",
         "The login screen takes ages to load before I can even enter my password."),
        ("Account access request stuck pending",
         "I submitted an access request last week and it still shows as pending with no update."),
    ],
    "Billing Issue": [
        ("Charged twice this month",
         "My card was charged two times for the same subscription plan this billing cycle."),
        ("Invoice amount does not match plan",
         "The invoice I received shows a higher amount than what my current plan should cost."),
        ("Refund not received yet",
         "I cancelled my subscription three weeks ago and the refund still hasn't shown up."),
        ("Unable to update payment method",
         "Every time I try to add a new credit card the payment form throws an error."),
        ("Coupon code not applied at checkout",
         "I entered a valid discount coupon but the final price did not change at all."),
        ("Billing page loads very slowly",
         "Every time I open the billing section the page takes forever before showing my invoices."),
        ("Would like an itemized billing breakdown",
         "It would help to see a more detailed breakdown of charges on my monthly invoice."),
    ],
    "Product Defect": [
        ("Export feature produces corrupted file",
         "Whenever I export my report to PDF, the downloaded file is corrupted and won't open."),
        ("App crashes when opening settings",
         "The mobile app crashes immediately every time I tap on the settings menu."),
        ("Data not saving after edit",
         "I edit a record, click save, but the changes disappear after refreshing the page."),
        ("Notifications stopped working",
         "I used to get email notifications for new comments but they suddenly stopped."),
        ("Broken image upload in profile",
         "Uploading a profile picture fails with a generic error message every single time."),
        ("Settings page freezes on load",
         "Opening the settings menu freezes the whole app and I have to force close it."),
        ("Report generation gets stuck",
         "Generating a report seems to hang indefinitely and never actually finishes."),
    ],
    "Performance Issue": [
        ("Dashboard takes forever to load",
         "The main dashboard takes over thirty seconds to load, it used to be instant."),
        ("Slow response during file upload",
         "Uploading large files causes the entire application to freeze for several minutes."),
        ("Search results lag significantly",
         "Typing a search query causes a long delay before any results appear on screen."),
        ("Frequent timeouts on peak hours",
         "During business hours the application times out repeatedly when loading reports."),
        ("Video streaming buffers constantly",
         "The in-app video tutorials buffer every few seconds even on a fast connection."),
        ("Application freezes randomly",
         "The whole application freezes at random moments and I have to restart it."),
        ("Login takes unusually long",
         "Signing in used to be instant but now it takes almost a minute to complete."),
    ],
    "Feature Request": [
        ("Requesting dark mode support",
         "It would be great if the application had a dark mode option for night time use."),
        ("Need bulk export option",
         "Please add a way to export multiple reports at once instead of one at a time."),
        ("Add calendar integration",
         "Could you integrate with Google Calendar so deadlines sync automatically."),
        ("Request for keyboard shortcuts",
         "Power users would benefit a lot from keyboard shortcuts for common actions."),
        ("Ability to customize dashboard widgets",
         "It would help to rearrange or hide dashboard widgets based on personal preference."),
        ("Suggestion to speed up report loading",
         "Reports could load faster if there was an option to load a lighter summary view."),
        ("Request for a cheaper billing tier",
         "It would be useful to have a lower cost plan for smaller teams like ours."),
    ],
}

URGENCY_LEVELS = ["Low", "Medium", "High", "Critical"]

VARIATIONS = [
    "", " This has been going on for two days now.", " Please help as soon as possible.",
    " I have already tried clearing my cache.", " This is affecting my whole team.",
    " Not sure if this is a known issue.", " Happens on both desktop and mobile.",
    " Please look into this soon.", " Let me know if you need more details.",
    " I contacted support before about a similar issue.", " Thanks in advance for your help.",
]


def maybe_swap_words(description):
    """Light randomization so identical templates don't repeat verbatim."""
    swaps = {"error": "issue", "problem": "trouble", "immediately": "right away"}
    for a, b in swaps.items():
        if a in description and random.random() < 0.3:
            description = description.replace(a, b, 1)
    return description


def build_rows(n_per_category=50):
    rows = []
    ticket_id = 1000
    for category, templates in CATEGORIES.items():
        for _ in range(n_per_category):
            subject, description = random.choice(templates)
            description = maybe_swap_words(description) + random.choice(VARIATIONS)
            rows.append({
                "ticket_id": ticket_id,
                "subject": subject,
                "description": description,
                "category": category,
                "urgency": random.choice(URGENCY_LEVELS),
            })
            ticket_id += 1
    random.shuffle(rows)

    # Simulate real-world annotation noise: a small fraction of tickets get
    # mislabeled, same as human agents would occasionally miscategorize tickets.
    all_categories = list(CATEGORIES.keys())
    noise_fraction = 0.06
    n_noisy = int(len(rows) * noise_fraction)
    for row in random.sample(rows, n_noisy):
        wrong_choices = [c for c in all_categories if c != row["category"]]
        row["category"] = random.choice(wrong_choices)

    return rows


if __name__ == "__main__":
    rows = build_rows(n_per_category=50)  # 5 categories x 50 = 250 tickets
    with open(config.DATA_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "subject", "description", "category", "urgency"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} tickets to {config.DATA_PATH}")
