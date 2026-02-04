# GIT configuration for distributing the stage a assignment template
REPO_SERVICE=github.com
REPO_TEMPLATE_ORG=CudaSudaWuda
# GITHUB CLASSROOM
GITHUB_CLASS_URL=https://classroom.github.com/classrooms/227265879-cs391-pmpp-f26

# Assignment specific values
NAME := $(shell basename ${PWD})

handoutfiles := handout/.vscode handout/hello.cu handout/Makefile

REPO_TEMPLATE=git@$(REPO_SERVICE):$(REPO_TEMPLATE_ORG)/$(NAME)

.DEFAULT_GOAL = all

.PHONY: all src handout template pushtemplate clean distclean

all: handout template

handout: ${handoutfiles}

handout/Makefile: Makefile.handout
	mkdir -p handout && cp $< $@

handout/hello.cu: hello.cu
	mkdir -p handout && cp $< $@

handout/.vscode: .vscode
	mkdir -p handout && cp -r $< $@

template: handout
	mkdir -p template/$(NAME) && cp -r handout/* handout/.vscode template/$(NAME)
	cd template/${NAME} && git init
	-cd template/${NAME} && git add -f . && git commit -m "initial commit" && git branch -M main
	-cd template/${NAME} && git remote add origin $(REPO_TEMPLATE)
	@echo "create repo $(NAME) at  https://$(REPO_SERVICE)/organizations/$(REPO_TEMPLATE_ORG)/repositories/new"
	@echo "then run make pushtemplate"

pushtemplate:
	cd template/${NAME} && git push -f -u origin main
	@echo "don't forget to visit $(GITHUB_CLASS_URL) to create assignment from the template"

clean:
	rm -rf $(wildcard handout *~)

distclean: clean
	rm -rf template
