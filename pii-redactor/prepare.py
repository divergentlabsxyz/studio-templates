"""The PII redactor's data: messages written from templates, every character
labelled, so nothing needs downloading.

    uv run prepare.py      # prints a few, and writes the test set to the cache

A message is a template with its slots filled: "Hi, this is {name}, my IBAN
is {iban}." Names, emails, phone numbers and account numbers are PII and are
masked; amounts, dates, order numbers, cities and companies look like them
and are not. The test set is written from templates and names that training
never sees, and a fifth of it is lowercased, so a redactor that learned the
words rather than the context leaks.
"""

import json
import random
import re
import string

from studio import cache_dir

KINDS = ["O", "NAME", "EMAIL", "PHONE", "ACCOUNT"]
SLOT_KIND = {
    "name": "NAME", "first": "NAME", "last": "NAME", "email": "EMAIL", "phone": "PHONE",
    "iban": "ACCOUNT", "card": "ACCOUNT", "account": "ACCOUNT",
}  # fmt: skip
CACHE = cache_dir("pii-redactor")
TEST_SIZE = 2000
VALIDATION_SIZE = 500
TEST_LOWERCASE = 0.2

FIRST = """Emma Liam Sofia Noah Mila Lucas Yara Daan Anna Mohammed Fatima Wei Jun Priya Arjun
Aisha Kofi Amara Mateo Lucia Sven Ingrid Olga Dmitri Hana Kenji Yuki Chloe Hugo Zoe Felix
Nora Omar Leila Tomasz Kasia Pedro Ines Rafael Bea Siobhan Declan Thandiwe Sipho Ngozi
Chidi Elif Emre Anika Rohan Mei Tariq Samira Jonas Lotte Bram Femke Sanne Joost Marta
Paulo Camila Diego Valentina Akira Sakura Min-jun Ji-woo Andrei Ioana Petra Milan Nadia
Yusuf Zainab Ravi Deepa Ewa Jakub Astrid Magnus Freya Oskar Ilse Wouter Maud Thijs Esra
Hamza Imane Rania Karim Lior Noa Aditi Vikram Grace Ethan Isla Oliver Amelia Harry Ruby
Jack Ava George Lily Leo Freddie Ivy Arthur Ada""".split()

LAST = """de Vries Jansen Bakker Visser Smit Meijer Mulder Bos Vos Peters Hendriks Dekker
Brouwer Dijkstra Kowalski Nowak Wiśniewska Müller Schmidt Schneider Fischer Weber Becker
Rossi Russo Bianchi Romano Garcia Martinez Lopez Gonzalez Silva Santos Oliveira Costa
Pereira Ferreira Dubois Laurent Moreau Lefebvre Leroy Nguyen Tran Le Pham Kim Lee Park
Choi Wang Li Zhang Liu Chen Yang Huang Tanaka Suzuki Takahashi Watanabe Ito Sato Patel
Shah Singh Kumar Sharma Gupta Khan Ahmed Hassan Ali Ibrahim Mensah Okafor Adeyemi Nwosu
Dlamini Ndlovu Mokoena Yilmaz Kaya Demir Sahin Andersson Johansson Karlsson Nilsson
Hansen Johansen Olsen Murphy Kelly O'Brien Walsh Smith Jones Taylor Brown Williams Wilson
Evans Thomas Roberts Cohen Levi Mizrahi Popescu Ionescu Novak Horvat Ivanova Petrov""".split()

COMPANIES = ["Apple", "Revolut", "Bol", "Zalando", "Albert Heijn", "Spotify", "Booking",
    "Uber", "Amazon", "Coolblue", "IKEA", "Adyen", "Mollie", "Netflix", "Vodafone", "KLM"]  # fmt: skip
CITIES = ["Amsterdam", "Rotterdam", "Utrecht", "Berlin", "Paris", "Lisbon", "Madrid", "London",
    "Dublin", "Warsaw", "Milan", "Lagos", "Nairobi", "Tokyo", "Seoul", "Toronto", "Austin"]  # fmt: skip
DOMAINS = ["gmail.com", "outlook.com", "proton.me", "icloud.com", "hey.com", "kpn.nl",
    "example.org", "acme.io", "uni-utrecht.nl", "yahoo.co.uk"]  # fmt: skip
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December"]  # fmt: skip

TRAIN_TEMPLATES = """
Hi, this is {name}. My card {card} was charged twice on {date}.
Please refund {amount} to {iban}, the account holder is {name}.
My name is {first} {last} and my email is {email}.
You can reach me on {phone} after {time}.
{name} called on {weekday} about order {order}.
Dear {first}, your payment of {amount} to {company} has been received.
Can you update my phone number to {phone}? Thanks, {first}
Transfer {amount} from account {account} to {iban} please.
The customer, {name}, says the delivery to {city} is late.
Email {email} or call {phone} if anything changes.
Order {order} was shipped to {city} on {date}.
Hello {first}, we could not verify your IBAN {iban}.
{first} {last} ({email}) asked to close account {account}.
I paid {amount} at {company} with card {card} but it was declined.
Ticket from {email}: my card {card} is blocked since {weekday}.
Please send the invoice to {name}, {email}.
Hi team, {first} here. My new number is {phone}.
The refund of {amount} for order {order} goes to {iban}.
{name} from {city} reports a charge of {amount} by {company}.
Kind regards, {name} | {phone} | {email}
Could you check why {company} charged {amount} on {date}?
My account number is {account} and my name is {name}.
We spoke to {first} on {date} at {time} about {order}.
Is {iban} still the right account for {first} {last}?
Thanks {first}! I've forwarded this to {email}.
Card {card} belongs to {name}, who lives in {city}.
Reminder: {first} owes {amount} for the trip to {city}.
New contact: {name}, {phone}, works at {company}.
The {weekday} standup moved to {time}, ask {first} for details.
Please confirm the transfer of {amount} to {name} ({iban}).
Hey, it's {first}. Order {order} arrived broken.
Your verification code was sent to {phone} on {date}.
Account {account} was opened by {first} {last} in {city}.
{first} wrote from {email} at {time} on {weekday}.
Can {first} call me back on {phone}? It's about {order}.
I moved to {city}; my email is now {email}.
Please block card {card}, I lost it in {city} on {weekday}.
{name} paid {amount} into {account} on {date}.
Write to {email} with the order number {order}.
Thanks for your help, {name}.
""".strip().splitlines()

TEST_TEMPLATES = """
Good morning, {first} {last} speaking, could someone ring me at {phone}?
Following up on my chat with {first}: the money ({amount}) should go to {iban}.
Signed, {name}. Contact: {email}.
Why did {company} take {amount} from card {card} on {date}?
Forwarding this from {first}, whose number is {phone}, re: order {order}.
Our records show {name} as the owner of account {account}.
Ping {email} once the {weekday} delivery to {city} is confirmed.
hello my name is {first} and my bank account is {iban}
{last}, {first} - {phone} - {city}
I never authorised the {amount} payment on card {card}, regards {name}
Please add {email} to the account of {first} {last}.
At {time} on {date}, {name} reported order {order} missing.
Account {account} ({name}) is overdrawn by {amount}.
Best, {first} {last}, sent from {city}
{name} would like a call on {phone} before {weekday}.
Could you move {amount} from {account} to {first}'s account {iban}?
""".strip().splitlines()


def split_names(names, every=4):
    """Train on three in four, test on the rest, so test names are new."""
    return [n for i, n in enumerate(names) if i % every], names[::every]


TRAIN_FIRST, TEST_FIRST = split_names(FIRST)
TRAIN_LAST, TEST_LAST = split_names(LAST)


def digits(rng, n):
    return "".join(rng.choice(string.digits) for _ in range(n))


def fill(rng, slot, first, last):
    if slot == "name":
        return f"{first} {last}"
    if slot == "first":
        return first
    if slot == "last":
        return last
    if slot == "email":
        user = rng.choice(
            [f"{first}.{last}", f"{first[0]}{last}", f"{first}{digits(rng, 2)}", f"{last}.{first[0]}"]
        )
        user = re.sub(r"[^a-z0-9.]", "", user.lower()) or "info"
        return f"{user}@{rng.choice(DOMAINS)}"
    if slot == "phone":
        return rng.choice(
            [
                f"+31 6 {digits(rng, 4)} {digits(rng, 4)}",
                f"06-{digits(rng, 8)}",
                f"({digits(rng, 3)}) {digits(rng, 3)}-{digits(rng, 4)}",
                f"+44 7{digits(rng, 3)} {digits(rng, 6)}",
                f"020 {digits(rng, 3)} {digits(rng, 4)}",
            ]
        )
    if slot == "iban":
        country = rng.choice(["NL", "DE", "GB", "FR", "BE", "ES"])
        bank = "".join(rng.choice(string.ascii_uppercase) for _ in range(4))
        body = country + digits(rng, 2) + bank + digits(rng, 10)
        return rng.choice([" ", ""]).join(body[i : i + 4] for i in range(0, len(body), 4))
    if slot == "card":
        number = digits(rng, 16)
        return rng.choice([" ", "-", ""]).join(number[i : i + 4] for i in range(0, 16, 4))
    if slot == "account":
        return digits(rng, rng.choice([8, 9, 10]))
    if slot == "amount":
        whole = rng.choice([str(rng.randint(1, 99)), f"{rng.randint(1, 9)},{digits(rng, 3)}"])
        return rng.choice(["€", "$", "£", "EUR "]) + whole + rng.choice(["", f".{digits(rng, 2)}"])
    if slot == "date":
        return rng.choice(
            [
                f"{rng.randint(1, 28)} {rng.choice(MONTHS)}",
                f"2026-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}",
                f"{rng.randint(1, 28)}/{rng.randint(1, 12)}",
            ]
        )
    if slot == "time":
        return f"{rng.randint(7, 21)}:{rng.choice(['00', '15', '30', '45'])}"
    if slot == "order":
        return rng.choice([f"#{digits(rng, 6)}", f"A-{digits(rng, 5)}", f"{digits(rng, 4)}-{digits(rng, 3)}"])
    if slot == "weekday":
        return rng.choice(WEEKDAYS)
    if slot == "city":
        return rng.choice(CITIES)
    if slot == "company":
        return rng.choice(COMPANIES)
    raise ValueError(f"unknown slot {slot}")


def message(rng, templates, firsts, lasts, lowercase=0.0):
    """One message and a kind per character, as indices into KINDS."""
    template = rng.choice(templates)
    first, last = rng.choice(firsts), rng.choice(lasts)
    text, labels = "", []
    for piece in re.split(r"(\{\w+\})", template):
        if piece.startswith("{"):
            slot = piece[1:-1]
            value = fill(rng, slot, first, last)
            kind = KINDS.index(SLOT_KIND.get(slot, "O"))
        else:
            value, kind = piece, 0
        text += value
        labels += [kind] * len(value)
    if rng.random() < lowercase:
        text = text.lower()
    return text, labels


def train_message(rng, lowercase=0.0):
    return message(rng, TRAIN_TEMPLATES, TRAIN_FIRST, TRAIN_LAST, lowercase)


def entities(labels):
    """(start, end, kind) for each run of one kind that is not O."""
    spans, start = [], None
    for i, kind in enumerate(labels + [0]):
        if start is not None and kind != labels[start]:
            spans.append((start, i, labels[start]))
            start = None
        if start is None and kind != 0 and i < len(labels):
            start = i
    return spans


def load():
    """validation and test, each a list of (text, labels), the same in every
    run. Both are written from the test templates and names, which training
    never sees: validation is a smaller draw that runs chart as they go."""
    path = CACHE / "test-v2.jsonl"
    if not path.exists():
        rng = random.Random(1234)
        rows = [
            message(rng, TEST_TEMPLATES, TEST_FIRST, TEST_LAST, TEST_LOWERCASE)
            for _ in range(TEST_SIZE)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    test = [tuple(json.loads(line)) for line in path.read_text().splitlines()]
    rng = random.Random(99)
    validation = [
        message(rng, TEST_TEMPLATES, TEST_FIRST, TEST_LAST, TEST_LOWERCASE)
        for _ in range(VALIDATION_SIZE)
    ]
    return validation, test


if __name__ == "__main__":
    validation, test = load()
    rng = random.Random(0)
    for _ in range(4):
        text, labels = train_message(rng)
        marks = "".join(" " if k == 0 else KINDS[k][0] for k in labels)
        print(text)
        print(marks)
    count = sum(len(entities(labels)) for _, labels in test)
    print(f"Test: {len(test)} messages, {count} entities, at {CACHE}")
