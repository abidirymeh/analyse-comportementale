import uuid

class TrackedPerson:
    def __init__(self, id, landmarks):
        self.id = id
        self.landmarks = landmarks 
        self.age = 0

class PersonTracker:
    def __init__(self):
        self.people = {}
        self.next_id = 0

    def update(self, results, frame):
        tracked = []
        if results.pose_landmarks:
            lm = results.pose_landmarks
            person_id = self.next_id
            self.next_id += 1

            person = TrackedPerson(person_id, results)
            person.age = 0
            self.people[person_id] = person
            tracked.append(person)

        for pid in list(self.people.keys()):
            self.people[pid].age += 1
            if self.people[pid].age < 100:  
                tracked.append(self.people[pid])

        return tracked