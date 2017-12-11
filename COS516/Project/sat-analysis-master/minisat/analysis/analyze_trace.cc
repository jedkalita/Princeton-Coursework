#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#include "core/SolverTypes.h"

using namespace std;
using namespace Minisat;


#ifndef NDEBUG
//#define DEBUG_EXTRA
#endif

enum skip {
  UNKNOWN = 0,
  SKIP,
  NO_SKIP,
};


/*
 *  Abstract classes
 */

class Event {
  protected:
    bool required;
  public:
    Event(bool init_required) : required(init_required) { }
    bool is_required() { return required; }
    virtual bool is_skippable() { return false; }
    void set_required() { if (!required) { required = true; handle_required(); } }
    virtual void print(ostream& out) = 0;
  private:
    virtual void handle_required() { }
};

class Assignment : public Event {
  protected:
    Lit lit;
  public:
    Assignment(Lit l) : Event(false), lit(l) { }
    Assignment(istream& in) ;
    Lit get_lit() { return lit; }
    unsigned get_var() { return var(lit); }
    bool get_sign() { return sign(lit); }
};

class ClauseAddition : public Event {
  protected:
    vector<Lit> lits;
    int deletion_time;
  public:
    ClauseAddition() : Event(false), deletion_time(-1) { }  // Only for when empty clause is learned
    ClauseAddition(istream& in) ;
    vector<Lit>& get_lits() { return lits; }
    void set_deleted() ;
    virtual void print(ostream& out) ;  // override
};


/*
 * Concrete classes
 */

class Branch : public Assignment {
  public:
    Branch(istream& in) : Assignment(in) { }
    bool is_skippable() { return !required; } // override
    void print(ostream& out) ;  // override
};

class Implication : public Assignment {
  protected:
    ClauseAddition* antecedent;  // 0 if antecedent is unit clause
    Assignment** required_assignments;  // null-terminated array of dependencies
    enum skip skippable;
  public:
    Implication(Lit lit, ClauseAddition* ante) ;
    bool is_skippable() ;  // override
    void print(ostream& out) ;  // override
  private:
    void handle_required() ;  // override
};

class OriginalClause : public ClauseAddition {
  public:
    OriginalClause(istream& in) : ClauseAddition(in) { }
};

class LearnedClause : public ClauseAddition {
  protected:
    ClauseAddition& antecedent;
    Assignment** required_assignments;  // null-terminated array of dependencies
    enum skip skippable;
  public:
    LearnedClause(istream& in, ClauseAddition& conflicting_clause) ;
    LearnedClause(ClauseAddition& conflicting_clause) ;  // Only for when empty clause is learned
    bool is_skippable() ;  // override
  private:
    void handle_required() ;  // override
};

class Reset : public Event {
  public:
    Reset() : Event(true) { }
    void print(ostream& out) { out << "r\n"; }  // override
};

class Deletion : public Event {
  protected:
    ClauseAddition* clause;
  public:
    Deletion(istream& in) ;
    void print(ostream& out) { out << "d "; clause->print(out); }  // override
};

class AssignmentComplete : public Event {
  protected:
    Assignment** final_assignments;  // null-terminated array
  public:
    AssignmentComplete() ;
    void print(ostream& out) ;  // override
    void handle_required() ;  // override
};


/*
 * Global variables
 */

vector<Event*> event_list;  // list of events in order from trace
vector<ClauseAddition*> clauses;  // mapping of clause ID's to clause addition events
vector<Assignment*> assignments;  // mapping of variables to assignments
vector<Assignment*> trail;  // Stack of currently relevant assignments
vector<unsigned> trail_lim;  // Stack of trail indices for branches
unsigned deletion_count;


/*
 *  Functions
 */

void print_lit(ostream& out, Lit lit) { out << (sign(lit) ? "-" : "") << var(lit); }

Lit read_lit(istream& in) {
  int parsed_lit = 0;
  in >> parsed_lit;
  assert (parsed_lit != 0);
  return mkLit(abs(parsed_lit), parsed_lit < 0); 
}


ClauseAddition::ClauseAddition(istream& in) : Event(false), deletion_time(-1) {
  while (true) {
    int parsed_lit = 0;
    in >> parsed_lit;
    if (parsed_lit == 0) break;
    lits.push_back(mkLit(abs(parsed_lit), parsed_lit < 0));
  }
}

void ClauseAddition::print(ostream& out) {
  for (vector<Lit>::iterator it = lits.begin(); it != lits.end(); ++it) {
    print_lit(out, *it);
    out << " ";
  }
  out << "0 ";
  if (deletion_time >= 0)
    out << deletion_time;
  out << endl;
}

void ClauseAddition::set_deleted() {
  deletion_time = deletion_count++;
}

Deletion::Deletion(istream& in) : Event(true) {
  int clause_id = -1;
  in >> clause_id;
  assert(clause_id >= 0 && (unsigned)clause_id < clauses.size());
  clause = clauses[clause_id];
  assert(clause != 0);
  clauses[clause_id] = 0;

  clause->set_deleted();
}


Assignment::Assignment(istream& in) : Event(false) {
  lit = read_lit(in);
}


void Branch::print(ostream& out) {
  out << "b ";
  print_lit(out, lit);
  out << "\n";
}


Implication::Implication(Lit lit, ClauseAddition* ante) : Assignment(lit), antecedent(ante), skippable(UNKNOWN) {
  if (antecedent == 0) {
    required_assignments = 0;
  } else {
#ifdef DEBUG_EXTRA
    cout << "Initializing implicaton of ";
    print_lit(cout, lit);
    cout << ". Antecedent is ";
    antecedent->print(cout);
#endif
    vector<Lit>& lits = antecedent->get_lits();
    required_assignments = new Assignment*[lits.size()];
    unsigned i = 0;
    for (vector<Lit>::iterator it = lits.begin(); it != lits.end(); ++it) {
      unsigned v = var(*it);
      if ((int)v != var(lit)) {
        assert(v < assignments.size());
        Assignment* assignment = assignments[v];
        assert(assignment != 0);
        assert(assignment->get_sign() != sign(*it));
        required_assignments[i++] = assignment;
      }
    }
    assert(i == lits.size() - 1);
    required_assignments[i] = 0;
  }
}

void Implication::handle_required() {
#ifdef DEBUG_EXTRA
  cout << "Implication required: ";
  print(cout);
#endif
  required = true;
  if (antecedent != 0) {
    antecedent->set_required();
    Assignment** required_assignment = required_assignments;
    while (*required_assignment != 0) {
      (*required_assignment)->set_required();
      ++required_assignment;
    }
  }
}

bool Implication::is_skippable() {
  if (skippable == SKIP)
    return true;
  else if (skippable == NO_SKIP)
    return false;

  if (required)
    skippable = NO_SKIP;
  else if (antecedent == 0)
    skippable = SKIP;
  else if (antecedent->is_skippable())
    skippable = SKIP;
  else {
    skippable = NO_SKIP;
    Assignment** required_assignment = required_assignments;
    while (*required_assignment != 0) {
      if ((*required_assignment)->is_skippable()) {
        skippable = SKIP;
        break;
      }
      ++required_assignment;
    }
  }
  return (skippable == SKIP);
}

void Implication::print(ostream& out) {
  out << "i ";
  print_lit(out, lit);
  out << "\n";
}


LearnedClause::LearnedClause(istream& in, ClauseAddition& conflicting_clause) : ClauseAddition(in), antecedent(conflicting_clause), skippable(UNKNOWN) {
  vector<Lit>& lits = antecedent.get_lits();
  required_assignments = new Assignment*[lits.size() + 1];
  unsigned i = 0;
  for (vector<Lit>::iterator it = lits.begin(); it != lits.end(); ++it) {
    Assignment* assignment = assignments[var(*it)];
    assert(assignment->get_sign() != sign(*it));
    required_assignments[i++] = assignment;
  }
  assert(i == lits.size());
  required_assignments[i] = 0;
}

LearnedClause::LearnedClause(ClauseAddition& conflicting_clause) : antecedent(conflicting_clause) {
  vector<Lit>& lits = antecedent.get_lits();
  required_assignments = new Assignment*[lits.size() + 1];
  unsigned i = 0;
  for (vector<Lit>::iterator it = lits.begin(); it != lits.end(); ++it) {
    Assignment* assignment = assignments[var(*it)];
    assert(assignment->get_sign() != sign(*it));
    required_assignments[i++] = assignment;
  }
  assert(i == lits.size());
  required_assignments[i] = 0;
}

void LearnedClause::handle_required() {
#ifdef DEBUG_EXTRA
  cout << "LearnedClause required: ";
  print(cout);
#endif
  required = true;
  antecedent.set_required();
  Assignment** required_assignment = required_assignments;
  while (*required_assignment != 0) {
    (*required_assignment)->set_required();
    ++required_assignment;
  }
}

bool LearnedClause::is_skippable() {
  if (skippable == SKIP)
    return true;
  else if (skippable == NO_SKIP)
    return false;

  if (required)
    skippable = NO_SKIP;
  else if (antecedent.is_skippable())
    skippable = SKIP;
  else {
    skippable = NO_SKIP;
    Assignment** required_assignment = required_assignments;
    while (*required_assignment != 0) {
      if ((*required_assignment)->is_skippable()) {
        skippable = SKIP;
        break;
      }
      ++required_assignment;
    }
  }
  return (skippable == SKIP);
}


AssignmentComplete::AssignmentComplete() : Event(false) {
  unsigned i, highest_var = assignments.size() - 1;
  final_assignments = new Assignment*[highest_var + 1]; 
  for (i = 1; i <= highest_var; i++) {
    assert(assignments[i] != 0);
    final_assignments[i-1] = assignments[i];
  }
  final_assignments[highest_var] = 0;
}

void AssignmentComplete::handle_required() {
  for (Assignment** required_assignment = final_assignments;
       *required_assignment != 0;
       required_assignment++) {
    (*required_assignment)->set_required();
  }
}

void AssignmentComplete::print(ostream& out) {
  out << "SAT" << endl;
  for (Assignment** a = final_assignments; *a != 0; a++) {
    (*a)->print(out);
  }
  out << "SAT" << endl;
}


void add_assignment(Assignment& assignment) {
  unsigned v = assignment.get_var();
  if (v < assignments.size()) {
    assert(assignments[v] == 0);
  } else {
    assignments.resize(v + 1);
  }
  assignments[v] = &assignment;
  trail.push_back(&assignment);
  event_list.push_back(&assignment);
#ifdef DEBUG_EXTRA
  cout << "NEW assignment: ";
  print_lit(cout, assignment.get_lit());
  cout << endl;
#endif
}

void add_clause(ClauseAddition& clause, int id = -1) {
  if (id >= 0) {
    if ((unsigned)id < clauses.size()) {
      assert(clauses[id] == 0);
    } else {
      clauses.resize(id + 1);
    }
    clauses[id] = &clause;
  }
  event_list.push_back(&clause);
#ifdef DEBUG_EXTRA
  cout << "new clause: ";
  clause.print(cout);
  cout << flush;
#endif
}

unsigned decision_level() {
  return trail_lim.size();
}

void new_decision_level() {
  trail_lim.push_back(trail.size());
}

void backtrack_to(unsigned level) {
  assert(level <= decision_level());
  if (level >= decision_level())
    return;

  unsigned i;
  for (i = trail_lim[level]; i < trail.size(); ++i) {
    assignments[trail[i]->get_var()] = 0;
  }
  trail.resize(trail_lim[level]);
  trail_lim.resize(level);
}

LearnedClause* parse_conflict(istream& in) {
  int conflicting_id, backtrack_level;
  in >> conflicting_id >> backtrack_level;

  ClauseAddition* conflicting_clause = clauses[conflicting_id];
  assert(conflicting_clause != 0);

#ifdef DEBUG_EXTRA
  cout << "confict: ";
  conflicting_clause->print(cout);
  cout << "backtracking to level " << backtrack_level << endl;
#endif

  if (backtrack_level == -1) {
    LearnedClause* clause = new LearnedClause(*conflicting_clause);
    add_clause(*clause);
    return clause;
  }

  int id = -1;
  in >> ws;
  if ('0' <= in.peek() && in.peek() <= '9') {
    in >> id;
  }
  in.ignore(1, ':');
  LearnedClause* clause = new LearnedClause(in, *conflicting_clause);
  add_clause(*clause, id);

  backtrack_to(backtrack_level);

  // Learned clause is asserting, first literal is implied
  Implication* implication = new Implication(clause->get_lits()[0], clause);
  add_assignment(*implication);

  return 0;
}

void parse_implication(istream& in) {
  Lit lit = read_lit(in);
  int antecedent_id;
  in >> antecedent_id;
  assert(antecedent_id >= 0 && (unsigned)antecedent_id < clauses.size());
  ClauseAddition* antecedent = clauses[antecedent_id];
  assert(antecedent != 0);
  Implication* implication = new Implication(lit, antecedent);
  add_assignment(*implication);
}

void parse_clause_movement(istream& in) {
  vector<ClauseAddition*> new_clauses;
  new_clauses.reserve(clauses.size() / 4u);
  while (true) {
    unsigned old_id, new_id;
    in >> old_id >> new_id;
    assert (old_id < clauses.size());
    if (new_id >= new_clauses.size())
      new_clauses.resize(new_id + 1);
    assert(clauses[old_id] != 0);
    new_clauses[new_id] = clauses[old_id];
    in >> ws;
    if (in.peek() != 'm')
      break;
    in.ignore(1, 'm');
  }
  clauses.swap(new_clauses);
}

Event& parse(istream& in) {

  Event* final_event = 0;
  while (final_event == 0) {
    in >> ws;
    if ('0' <= in.peek() && in.peek() <= '9') {
      unsigned clause_id;
      in >> clause_id;
      in.ignore(1, ':');
      OriginalClause* clause = new OriginalClause(in);
      add_clause(*clause, clause_id);
      assert(clause->get_lits().size() > 1u);
    } else {
      char c = '\a';
      in >> c;
      switch (c) {
      case ':': {
        OriginalClause* clause = new OriginalClause(in);
        add_clause(*clause);
        unsigned clause_size = clause->get_lits().size();
        if (clause_size == 0u) {
          final_event = clause;
        } else {
          assert(clause_size == 1u);
          add_assignment(*new Implication(clause->get_lits()[0], clause));
        }
      } break;
      case 'i':
        parse_implication(in);
        break;
      case 'b':
        new_decision_level();
        add_assignment(*new Branch(in));
        break;
      case 'k':
        final_event = parse_conflict(in);
        break;
      case 'r':
        backtrack_to(0);
        break;
      case 'd':
        event_list.push_back(new Deletion(in));
        break;
      case 'm':
        parse_clause_movement(in);
        break;
      case '\a':
        cerr << "End of file without finding empty clause?" << endl;
        exit(EXIT_FAILURE);
        break;
      case 'S':
        final_event = new AssignmentComplete();
        event_list.push_back(final_event);
        break;
      case 'U':
        cerr << "Early UNSAT?! Aborting..." << endl;
        exit(EXIT_FAILURE);
        break;
      default:
        cerr << "Parse error! " << c << endl;
        exit(EXIT_FAILURE);
        break;
      }
    }
  }
  return *final_event;
}

int main(int argc, char* argv[]) {
  int events = 0;
  int required = 0;
  int skippable = 0;
  int branches = 0;
  int req_branches = 0;
  int skip_branches = 0;
  int props = 0;
  int req_props = 0;
  int skip_props = 0;

  if (argc <= 2) {
    cerr << "usage: " << argv[0] << " INPUT_FILE OUTPUT_FILE" << endl;
    exit(EXIT_FAILURE);
  }

  cout << "Parsing..." << endl;

  ifstream in(argv[1]);
  Event &final_event = parse(in);
  in.close();

  cout << "Traversing dependencies..." << endl;

  // Recursively find all critical work.
  final_event.set_required();

  // Now we look through all events to collect statistics.
  for (vector<Event*>::iterator it = event_list.begin(); it != event_list.end(); ++it) {
    ++events;
    if ((*it)->is_required()) {
      ++required;
      if (NULL != dynamic_cast<Branch*>(*it)) {
        ++req_branches;
        ++branches;
      }
      else if (NULL != dynamic_cast<Implication*>(*it)) {
        ++req_props;
        ++props;
      }
    }
    else if ((*it)->is_skippable()) {
      ++skippable;
      if (NULL != dynamic_cast<Branch*>(*it)) {
        ++skip_branches;
        ++branches;
      }
      else if (NULL != dynamic_cast<Implication*>(*it)) {
        ++skip_props;
        ++props;
      }
    }
    else {
      if (NULL != dynamic_cast<Branch*>(*it)) ++branches;
      else if (NULL != dynamic_cast<Implication*>(*it)) ++props;
    }
  }

  cout << "Writing out analysis..." << endl;

  ofstream out(argv[2]);

  if (0) {
    // Print out entire annotated trace
    for (vector<Event*>::iterator it = event_list.begin(); it != event_list.end(); ++it) {
      if ((*it)->is_skippable()) {
        out << "~ ";
      }
      else if ((*it)->is_required()) {
        out << "! ";
      }
      (*it)->print(out);
    }
  }

  // Print out statistics
  out << argv[1] << ',';
  out << events << ',';
  out << required << ',';
  out << skippable << ',';
  out << branches << ',';
  out << req_branches << ',';
  out << skip_branches << ',';
  out << props << ',';
  out << req_props << ',';
  out << skip_props << endl;

  out.close();

  cout << "Done." << endl;
}
