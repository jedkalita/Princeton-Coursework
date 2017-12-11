#include <assert.h>
/** prototypes */
int l;
int nondet_int ();
void lock (void) {l = 1; };
void unlock (void) {l = 0;};
//int l = 0;

int read_input (void) { return nondet_int (); }

/** FUNCTION lock1 */
void lock1 (void)
{
    int in_irq;
    int buf[10];
    int i;
    
    if (in_irq)
    {
        //l++;
        lock ();
        //l++;
        assert(l != 2);
    }
    
    for (i = 0; i < 5; i++)
        buf [i] = read_input ();
    
    if (in_irq)
    {
        //l++;
        lock ();
        //l++;
        assert(l != 2);
    }
}

/** FUNCTION lock2 */
void lock2 (void)
{
    int request, old, total;
    
    int lock_before_loop = -1;
    int loop_count = 0;
    request = 0;
    
    //__CPROVER_assume(l = 0);
    do {
        if (l == 1 && loop_count == 0)
        {
            lock_before_loop = 1;
        }
        else if (l == 0 && loop_count == 0)
        {
            lock_before_loop = 0;
        }
        lock ();
        //assert(l != 2);
        old = total;
        request = read_input ();
        if (request)
        {
            unlock ();
            //assert(l != -1);
            total = total + 1;
        }
        loop_count++;
    } while (total != old);
    assert ( (lock_before_loop == 0 && l == 1 && total == old) || (lock_before_lo\
                                                                   op != 0));
    //assert ( (lock_before_loop == 1 && total == old) ||
    unlock ();
    //assert(l != -1);                                                            
}

